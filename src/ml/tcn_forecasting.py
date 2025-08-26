"""Temporal Convolutional Network (TCN) для прогнозирования вероятности дефекта и степени развития.

Вход: последовательности embedding'ов или агрегированных признаков (из Feature.extra['embedding'] + rms_*).
Выход: (p_defect_next, severity_next) в диапазоне 0..1.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import random
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from .utils import save_model, load_model
from src.database.models import Feature, RawSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_channels(cfg: str) -> List[int]:
    return [int(x) for x in cfg.split(',') if x.strip()]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

if nn is not None:
    class TCN(nn.Module):
        def __init__(self, input_dim: int, channels: List[int], kernel_size: int, dropout: float):
            super().__init__()
            layers: List[Any] = []
            prev = input_dim
            for i, ch in enumerate(channels):
                dilation = 2 ** i
                layers.append(TemporalBlock(prev, ch, kernel_size, dilation, dropout))
                prev = ch
            self.tcn = nn.Sequential(*layers)
            # Голова без финальной активации: первый выход — логит вероятности дефекта,
            # второй — непрерывная регрессионная величина (степень выраженности)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(prev, prev//2),
                nn.ReLU(),
                nn.Linear(prev//2, 2),  # [логит(p_defect), «сырая» степень]
            )
        def forward(self, x):  # x: (B,T,D) -> транспонируем в (B,D,T)
            x = x.transpose(1, 2)
            out = self.tcn(x)
            return self.head(out)
else:
    class TCN:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):  # pragma: no cover
            raise RuntimeError("torch/nn недоступны: установите extra 'ml-heavy' или пакет torch")

@dataclass
class TCNConfig:
    window_size: int
    horizon: int
    channels: List[int]
    kernel_size: int
    dropout: float
    max_epochs: int
    lr: float
    # SGD-параметры
    batch_size: int = 64
    seed: int = 42
    # Шедулер по валидационному лоссу
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-5
    # Логирование
    log_every: int = 5
    # Дополнительные параметры обучения
    val_fraction: float = 0.2  # доля валидации (последние окна по времени)
    early_stopping_patience: int = 10
    min_delta: float = 0.0
    # Генерация целевых меток
    target_method: str = "percentile"  # метод построения таргетов: 'percentile'
    target_percentile: float = 90.0     # перцентиль для порога дефекта


def _extract_feature_vector(f: Feature) -> Optional[List[float]]:
    emb = None
    if f.extra and isinstance(f.extra, dict):
        emb = f.extra.get('embedding')
    vec: List[float] = []
    if emb and isinstance(emb, list):
        try:
            vec = [float(x) for x in emb]
        except Exception:
            return None
    # добавим базовые агрегаты
    for attr in ['rms_a','rms_b','rms_c','mean_a','mean_b','mean_c']:
        v = getattr(f, attr, None)
        vec.append(float(v) if v is not None else 0.0)
    return vec if vec else None

async def load_sequence(
    session: AsyncSession,
    equipment_id,
    window: int,
    horizon: int,
    target_method: str = "percentile",
    target_percentile: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Формирует обучающие последовательности признаков и целевые метки для TCN.

    Назначение:
        - Загружает признаки для оборудования, собирает окна длиной ``window`` и таргеты на горизонт ``horizon``.
        - Порог для вероятности дефекта задаётся через метод/перцентиль.

    Параметры:
        - session: асинхронная сессия БД.
        - equipment_id: идентификатор оборудования.
        - window: длина окна по времени (T).
        - horizon: горизонт для вычисления таргета.
        - target_method: метод порога ('percentile').
        - target_percentile: значение перцентиля (0..100).

    Возвращает:
        - X: np.ndarray формы (N, T, D) — окна признаков.
        - y: np.ndarray формы (N, 2) — [p_defect, severity].

    Исключения:
        - ValueError: если данных недостаточно (меньше window+horizon+5 валидных векторов).
    """
    # Явно укажем условие соединения, чтобы не зависеть от настроек relationship
    q = (
        select(Feature)
        .join(RawSignal, Feature.raw_id == RawSignal.id)
        .where(RawSignal.equipment_id == equipment_id)
        .order_by(Feature.window_start.asc())
    )
    res = await session.execute(q)
    feats = res.scalars().all()
    vectors: List[List[float]] = []
    for f in feats:
        v = _extract_feature_vector(f)
        if v:
            vectors.append(v)
    if len(vectors) < window + horizon + 5:
        raise ValueError("Недостаточно данных для TCN")
    arr = np.array(vectors, dtype=np.float32)
    # Целевые значения: вероятность дефекта (эвристика) и степень выраженности (нормализованный рост среднего RMS)
    rms_idx_start = -6  # последние 6 значений - RMS/mean
    rms_triplets = arr[:, rms_idx_start:rms_idx_start+6]
    rms_mean = rms_triplets[:, :3].mean(axis=1)  # среднее по rms_a,b,c
    target_p = []
    target_s = []
    # Порог дефекта по методу и параметрам
    if target_method == "percentile":
        pct = float(target_percentile)
        pct = 0.0 if pct < 0.0 else (100.0 if pct > 100.0 else pct)
        thresh = np.percentile(rms_mean, pct)
    else:
        # Запасной вариант — стандартный 90-й перцентиль
        thresh = np.percentile(rms_mean, 90.0)
    for i in range(0, len(arr) - window - horizon):
        future_rms = rms_mean[i+window:i+window+horizon]
        p_defect = 1.0 if future_rms.mean() > thresh else 0.0
        # Степень: относительный рост к текущему среднему
        current = rms_mean[i+window-1]
        growth = max(future_rms.mean() - current, 0.0)
        severity = float(growth / (thresh + 1e-6))
        target_p.append(p_defect)
        target_s.append(min(1.0, severity))
    X_list = []
    for i in range(0, len(arr) - window - horizon):
        X_list.append(arr[i:i+window])
    X = np.stack(X_list)
    y = np.stack([target_p, target_s], axis=1)
    return X, y

async def train_tcn(equipment_id, config: Optional[TCNConfig] = None) -> Dict:
    """Обучает модель TCN для указанного оборудования и сохраняет артефакт.

    Поведение:
        - Делит данные на train/val по времени, нормализует по train, обучает с ранней остановкой и шедулером.
        - Сохраняет лучшие веса, конфигурацию и статистики нормализации в кеш моделей.

    Параметры:
        - equipment_id: идентификатор оборудования.
        - config: конфигурация обучения TCN (если None — берётся из настроек).

    Возвращает:
        - dict со статусом и метриками: {'status','samples','train_samples', 'validation_skipped', ...}
        - В случае нехватки данных: {'status': 'skipped', 'error': 'insufficient_data', ...}
        - В случае ошибки загрузки последовательности: {'status': 'error', 'error': 'sequence_load_failed', ...}

    Исключения:
        - RuntimeError: когда TCN отключён настройкой или отсутствует torch.
    """
    st = get_settings()
    if getattr(st, 'TCN_ENABLED', True) is False:
        raise RuntimeError("TCN отключен настройкой TCN_ENABLED=false")
    if torch is None or nn is None:
        raise RuntimeError("TCN недоступен: отсутствует torch (установите extra 'ml-heavy')")
    if config is None:
        config = TCNConfig(
            window_size=st.TCN_WINDOW_SIZE,
            horizon=st.TCN_PREDICTION_HORIZON,
            channels=_parse_channels(st.TCN_CHANNELS),
            kernel_size=st.TCN_KERNEL_SIZE,
            dropout=st.TCN_DROPOUT,
            max_epochs=st.TCN_MAX_EPOCHS,
            lr=st.TCN_LEARNING_RATE,
        )
    # Гарантируем наличие каталога для моделей
    try:
        model_dir = st.models_path / 'tcn'
        model_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Реально загрузим данные
    from src.database.connection import get_async_session
    try:
        async with get_async_session() as session:
            X, y = await load_sequence(
                session,
                equipment_id,
                config.window_size,
                config.horizon,
                config.target_method,
                config.target_percentile,
            )
    except ValueError as e:
        # Недостаточно данных для обучения
        return {
            'status': 'skipped',
            'error': 'insufficient_data',
            'required_min': int(config.window_size + config.horizon + 5),
            'detail': str(e),
        }
    except Exception as e:
        # Общая ошибка загрузки последовательности
        return {
            'status': 'error',
            'error': 'sequence_load_failed',
            'detail': str(e),
        }
    # Разбиение на train/val по времени (валидация — последние окна)
    N = X.shape[0]
    use_val = N >= 5 and config.val_fraction > 0.0
    if use_val:
        n_val = max(1, int(N * config.val_fraction))
        n_train = max(1, N - n_val)
        if n_train < 1:
            n_val = 1
            n_train = N - 1
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
    # Стандартизация только по train, применяем к val (если есть)
    try:
        x_flat = X_train.reshape(-1, X_train.shape[2])
        feat_mean = x_flat.mean(axis=0)
        feat_std = x_flat.std(axis=0)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
        X_train = (X_train - feat_mean.reshape(1, 1, -1)) / feat_std.reshape(1, 1, -1)
        if X_val is not None:
            X_val = (X_val - feat_mean.reshape(1, 1, -1)) / feat_std.reshape(1, 1, -1)
    except Exception:
        feat_mean = None
        feat_std = None
    # Сиды для воспроизводимости
    try:
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
    except Exception:
        pass
    device = torch.device('cpu')
    model = TCN(X.shape[2], config.channels, config.kernel_size, config.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    # Раздельные функции потерь: BCE для вероятности дефекта и MSE для непрерывной «severity»
    # Классификация обучается по логитам через BCEWithLogits
    crit_cls = nn.BCEWithLogitsLoss()
    crit_reg = nn.MSELoss()
    model.train()
    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device) if X_val is not None else None
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device) if y_val is not None else None
    # DataLoader для батчей
    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=max(1, int(config.batch_size)), shuffle=True)
    val_loader = None
    if X_val_t is not None and y_val_t is not None:
        val_ds = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_ds, batch_size=max(1, int(config.batch_size)), shuffle=False)
    # Шедулер снижения LR при плато по валидации
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr, verbose=False
    )
    best_state = None
    best_val = float('inf')
    best_epoch = -1
    no_improve = 0
    last_train_loss = None
    last_val_loss = None
    for epoch in range(config.max_epochs):
        # Train epoch с батчами
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            preds = model(xb)
            loss = crit_cls(preds[:, 0], yb[:, 0]) + crit_reg(preds[:, 1], yb[:, 1])
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            total_count += int(xb.size(0))
        last_train_loss = total_loss / max(1, total_count)
        # Валидация + шедулер + ранняя остановка
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    v_preds = model(xb)
                    v_loss = crit_cls(v_preds[:, 0], yb[:, 0]) + crit_reg(v_preds[:, 1], yb[:, 1])
                    val_total += float(v_loss.item()) * xb.size(0)
                    val_count += int(xb.size(0))
            v = val_total / max(1, val_count)
            last_val_loss = v
            scheduler.step(v)
            if v < (best_val - config.min_delta):
                best_val = v
                best_state = model.state_dict()
                best_epoch = epoch
                no_improve = 0
                if (epoch + 1) % max(1, config.log_every) == 0:
                    try:
                        cur_lr = opt.param_groups[0]['lr']
                    except Exception:
                        cur_lr = None
                    logger.info(f"Обучение TCN: эпоха={epoch+1}/{config.max_epochs} train_loss={last_train_loss:.6f} val_loss={v:.6f} (улучшение) lr={cur_lr}")
            else:
                no_improve += 1
                if no_improve >= config.early_stopping_patience:
                    logger.info(f"Ранняя остановка TCN на эпохе={epoch+1}: best_val={best_val:.6f} лучшая_эпоха={best_epoch}")
                    break
        else:
            # Если валидации нет — ориентируемся на train и можем шагать шедулером по train
            scheduler.step(last_train_loss)
        # Периодическое логирование, если не было ветки улучшения выше
        if val_loader is None and (epoch + 1) % max(1, config.log_every) == 0:
            try:
                cur_lr = opt.param_groups[0]['lr']
            except Exception:
                cur_lr = None
            logger.info(f"Обучение TCN: эпоха={epoch+1}/{config.max_epochs} train_loss={last_train_loss:.6f} lr={cur_lr}")
    # Сохраняем модель
    # Сохраняем модель и статистики нормализации (если рассчитаны)
    chosen_state = best_state if best_state is not None else model.state_dict()
    payload = {'state_dict': chosen_state, 'config': asdict(config), 'input_dim': X.shape[2]}
    if feat_mean is not None and feat_std is not None:
        payload['norm'] = {
            'mean': feat_mean.astype(np.float32).tolist(),
            'std': feat_std.astype(np.float32).tolist(),
        }
    save_model(payload, f"tcn/model_{equipment_id}")
    try:
        from src.utils.metrics import increment_counter
        increment_counter('model_cache_events_total', {'model_name': 'tcn', 'event': 'store'})
    except Exception:  # pragma: no cover
        pass
    result: Dict[str, Any] = {
        'status': 'trained',
        'samples': int(X.shape[0]),
        'train_samples': int(X_train.shape[0]),
        'validation_skipped': not use_val,
        'train_loss_last': last_train_loss if last_train_loss is not None else None,
    }
    if use_val:
        result.update({
            'val_samples': int(X_val.shape[0]) if X_val is not None else 0,
            'val_loss_best': None if best_val == float('inf') else best_val,
            'val_loss_last': last_val_loss,
            'best_epoch': None if best_epoch < 0 else best_epoch,
            'early_stopped': best_epoch >= 0 and no_improve >= config.early_stopping_patience,
        })
    return result

async def predict_tcn(equipment_id, recent_window: Optional[np.ndarray] = None, config: Optional[TCNConfig] = None) -> Dict:
    """Выполняет предсказание (p_defect_next, severity_next) для оборудования.

    Поведение:
        - Пытается загрузить модель из кеша; при отсутствии — обучает и повторно пытается загрузить.
        - Использует конфигурацию из артефакта для архитектуры и длины окна; нормализует входы по сохранённым mean/std.

    Параметры:
        - equipment_id: идентификатор оборудования.
        - recent_window: опциональное окно (T x D или 1 x T x D); если не задано, берём из БД.
        - config: конфигурация (флаги и параметры предсказания); используется как запасной вариант.

    Возвращает:
        - dict с ключами 'p_defect_next' и 'severity_next' в [0,1].
        - В негативных сценариях возвращает поля 'error'/'status' с деталями (например, torch_not_available,
            insufficient_data, model_not_available, state_dict_missing, state_load_failed и др.).
    """
    st = get_settings()
    # Фича-флаг отключения
    if getattr(st, 'TCN_ENABLED', True) is False:
        return {'p_defect_next': 0.0, 'severity_next': 0.0, 'disabled': True}
    if config is None:
        config = TCNConfig(
            window_size=st.TCN_WINDOW_SIZE,
            horizon=st.TCN_PREDICTION_HORIZON,
            channels=_parse_channels(st.TCN_CHANNELS),
            kernel_size=st.TCN_KERNEL_SIZE,
            dropout=st.TCN_DROPOUT,
            max_epochs=st.TCN_MAX_EPOCHS,
            lr=st.TCN_LEARNING_RATE,
        )
    if torch is None or nn is None:
        return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'torch_not_available'}
    data = load_model(f"tcn/model_{equipment_id}")
    if data is None:
        try:
            from src.utils.metrics import increment_counter
            increment_counter('model_cache_events_total', {'model_name': 'tcn', 'event': 'miss'})
        except Exception:  # pragma: no cover
            pass
        try:
            train_result = await train_tcn(equipment_id, config)
        except Exception as e:
            return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'train_failed', 'detail': str(e)}
        # Если обучение пропущено или завершилось ошибкой — возвращаем понятный ответ
        if isinstance(train_result, dict) and train_result.get('status') != 'trained':
            resp = {'p_defect_next': 0.0, 'severity_next': 0.0}
            if 'status' in train_result:
                resp['status'] = train_result['status']
            if 'error' in train_result:
                resp['error'] = train_result['error']
            if 'detail' in train_result:
                resp['detail'] = train_result['detail']
            if 'required_min' in train_result:
                resp['required_min'] = train_result['required_min']
            return resp
        data = load_model(f"tcn/model_{equipment_id}")
        if data is None:
            # Модель не была сохранена после обучения — возвращаем понятную ошибку
            return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'model_not_available'}
    else:
        try:
            from src.utils.metrics import increment_counter
            increment_counter('model_cache_events_total', {'model_name': 'tcn', 'event': 'hit'})
        except Exception:  # pragma: no cover
            pass
    # Проверяем наличие state_dict
    state = data.get('state_dict') if isinstance(data, dict) else None
    if state is None:
        return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'state_dict_missing'}
    # Корректное определение размерности входа
    input_dim = data.get('input_dim') if isinstance(data, dict) else None
    if input_dim is None:
        # Пытаемся определить по сохранённой нормализации
        try:
            norm = data.get('norm') if isinstance(data, dict) else None
            if norm and isinstance(norm.get('mean'), list):
                input_dim = int(len(norm['mean']))
        except Exception:
            input_dim = None
    if input_dim is None:
        # Пытаемся определить из recent_window, если он передан
        if recent_window is not None:
            try:
                arr = np.asarray(recent_window)
                if arr.ndim == 3:
                    input_dim = int(arr.shape[2])
                elif arr.ndim == 2:
                    input_dim = int(arr.shape[1])
                else:
                    return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'bad_window_shape'}
            except Exception as e:
                return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'input_dim_infer_failed', 'detail': str(e)}
        else:
            # Не можем надёжно определить input_dim — требуем recent_window или обновить артефакт модели
            return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'input_dim_unresolved'}
    # Используем сохранённую конфигурацию архитектуры (если есть)
    saved_conf = data.get('config') if isinstance(data, dict) else None
    if isinstance(saved_conf, dict):
        arch_channels = saved_conf.get('channels', config.channels)
        arch_kernel = int(saved_conf.get('kernel_size', config.kernel_size))
        arch_dropout = float(saved_conf.get('dropout', config.dropout))
        infer_window = int(saved_conf.get('window_size', config.window_size))
    else:
        arch_channels = config.channels
        arch_kernel = config.kernel_size
        arch_dropout = config.dropout
        infer_window = config.window_size
    model = TCN(input_dim=input_dim, channels=arch_channels, kernel_size=arch_kernel, dropout=arch_dropout)
    try:
        model.load_state_dict(state)  # type: ignore[arg-type]
    except Exception as e:
        return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'state_load_failed', 'detail': str(e)}
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    if recent_window is None:
        try:
            from src.database.connection import get_async_session
            async with get_async_session() as session:
                X_all, _ = await load_sequence(
                    session,
                    equipment_id,
                    infer_window,
                    config.horizon,
                    config.target_method,
                    config.target_percentile,
                )
            recent_window = X_all[-1:, :, :]
        except Exception as e:
            return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'recent_window_unavailable', 'detail': str(e)}
    else:
        # Валидация и приведение формы recent_window
        try:
            arr = np.asarray(recent_window, dtype=np.float32)
            if arr.ndim == 2:  # (T,D) -> (1,T,D)
                arr = arr.reshape(1, arr.shape[0], arr.shape[1])
            if arr.ndim != 3:
                return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'bad_window_shape'}
            # Длина окна: если длиннее сохранённого — обрежем справа; если короче, допускаем (TCN и AdaptiveAvgPool1d поддерживают переменную длину)
            if arr.shape[1] > infer_window:
                arr = arr[:, -infer_window:, :]
            elif arr.shape[1] < 1:
                return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'window_too_short', 'detail': f'need>=1, got={arr.shape[1]}'}
            elif arr.shape[1] < infer_window:
                try:
                    logger.warning(f"TCN predict: окно короче, чем в обучении (got={arr.shape[1]}, trained={infer_window}) — используем как есть")
                except Exception:
                    pass
            recent_window = arr
        except Exception as e:
            return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'window_preprocess_failed', 'detail': str(e)}
    # Если сохранены статистики нормализации — применяем их к recent_window
    try:
        norm = data.get('norm') if isinstance(data, dict) else None
        if norm and 'mean' in norm and 'std' in norm:
            mean = np.asarray(norm['mean'], dtype=np.float32)
            std = np.asarray(norm['std'], dtype=np.float32)
            std = np.where(std < 1e-8, 1.0, std)
            if mean.shape[0] == recent_window.shape[2]:
                recent_window = (recent_window - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    except Exception:
        # Не валим предсказание при проблеме нормализации
        pass
    X_t = torch.tensor(recent_window, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(X_t).cpu().numpy()[0]
    # Преобразуем логит в вероятность и ограничим severity в [0,1]
    p = 1.0 / (1.0 + np.exp(-float(out[0])))
    sev = float(np.clip(out[1], 0.0, 1.0))
    return {'p_defect_next': float(p), 'severity_next': float(sev)}

__all__ = ['TCNConfig', 'train_tcn', 'predict_tcn']
