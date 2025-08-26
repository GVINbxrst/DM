"""Temporal Convolutional Network (TCN) для прогнозирования вероятности дефекта и степени развития.

Вход: последовательности embedding'ов или агрегированных признаков (из Feature.extra['embedding'] + rms_*).
Выход: (p_defect_next, severity_next) в диапазоне 0..1.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
from typing import Any
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
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(prev, prev//2),
                nn.ReLU(),
                nn.Linear(prev//2, 2),  # p_defect, severity
                nn.Sigmoid()
            )
        def forward(self, x):  # x: (B,T,D) -> transpose to (B,D,T)
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
    # Дополнительные параметры обучения
    val_fraction: float = 0.2  # доля валидации (последние окна по времени)
    early_stopping_patience: int = 10
    min_delta: float = 0.0


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

async def load_sequence(session: AsyncSession, equipment_id, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    q = select(Feature).join(RawSignal).where(RawSignal.equipment_id == equipment_id).order_by(Feature.window_start.asc())
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
    # Целевые значения: вероятность дефекта (эвристика) и severity (нормализованный рост mean RMS)
    rms_idx_start = -6  # последние 6 значений - RMS/mean
    rms_triplets = arr[:, rms_idx_start:rms_idx_start+6]
    rms_mean = rms_triplets[:, :3].mean(axis=1)  # rms_a,b,c среднее
    target_p = []
    target_s = []
    thresh = np.percentile(rms_mean, 90)
    for i in range(0, len(arr) - window - horizon):
        future_rms = rms_mean[i+window:i+window+horizon]
        p_defect = 1.0 if future_rms.mean() > thresh else 0.0
        # severity: относительный рост к текущему среднему
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
    # Ensure model directory exists
    try:
        model_dir = st.models_path / 'tcn'
        model_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Реально загрузим данные
    from src.database.connection import get_async_session
    async with get_async_session() as session:
        X, y = await load_sequence(session, equipment_id, config.window_size, config.horizon)
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
    crit_cls = nn.BCELoss()
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
            else:
                no_improve += 1
                if no_improve >= config.early_stopping_patience:
                    break
        else:
            # Если валидации нет — ориентируемся на train и можем шагать шедулером по train
            scheduler.step(last_train_loss)
    # Сохраняем модель
    # Сохраняем модель и статистики нормализации (если рассчитаны)
    chosen_state = best_state if best_state is not None else model.state_dict()
    payload = {'state_dict': chosen_state, 'config': config.__dict__, 'input_dim': X.shape[2]}
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
            await train_tcn(equipment_id, config)
        except Exception as e:
            return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'train_failed', 'detail': str(e)}
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
    model = TCN( input_dim=input_dim, channels=config.channels, kernel_size=config.kernel_size, dropout=config.dropout)
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
                X_all, _ = await load_sequence(session, equipment_id, config.window_size, config.horizon)
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
            # Обеспечиваем длину окна window_size: берем последние window_size
            if arr.shape[1] >= config.window_size:
                arr = arr[:, -config.window_size:, :]
            else:
                return {'p_defect_next': 0.0, 'severity_next': 0.0, 'error': 'window_too_short'}
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
    return {'p_defect_next': float(out[0]), 'severity_next': float(out[1])}

__all__ = ['TCNConfig', 'train_tcn', 'predict_tcn']
