import torch
import torch.nn.functional as F
import pandas as pd


# -------------------------------------------------
# Parameter initialization
# -------------------------------------------------

def init_block(input_size, backcast_size, forecast_size, layers=4, layer_size=512, weight_scale=0.01):
    params = {}

    # First layer: input -> hidden
    params["W0"] = torch.randn(layer_size, input_size) * weight_scale
    params["b0"] = torch.zeros(layer_size)

    # Hidden layers
    for i in range(1, layers):
        params[f"W{i}"] = torch.randn(layer_size, layer_size) * weight_scale
        params[f"b{i}"] = torch.zeros(layer_size)

    # Theta head
    theta_size = backcast_size + forecast_size
    params["W_theta"] = torch.randn(theta_size, layer_size) * weight_scale
    params["b_theta"] = torch.zeros(theta_size)

    # Enable gradients (leaf tensors at init time)
    for k in params:
        params[k].requires_grad_()

    return params


# -------------------------------------------------
# Forward pass
# -------------------------------------------------

def nbeats_block_forward(x, params, layers=4):
    """
    x: [batch_size, input_size]
    returns backcast, forecast
    """
    h = x

    # Fully connected stack
    for i in range(layers):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        h = F.relu(F.linear(h, W, b))

    # Theta projection
    theta = F.linear(h, params["W_theta"], params["b_theta"])

    # Split theta
    backcast_size = params["W0"].shape[1]  # equals input_size
    backcast = theta[:, :backcast_size]
    forecast = theta[:, backcast_size:]

    return backcast, forecast


# -------------------------------------------------
# Data loading
# -------------------------------------------------

def load_passenger_series(csv_path, value_col):
    df = pd.read_csv(csv_path)

    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found. Available columns: {list(df.columns)}")

    series = torch.tensor(df[value_col].values, dtype=torch.float32)

    mean = float(series.mean())
    std = float(series.std() + 1e-8)
    series = (series - mean) / std

    return series, mean, std


def make_windows(series, input_size, forecast_size):
    X, Y = [], []
    T = series.shape[0]
    n = T - input_size - forecast_size + 1
    if n <= 0:
        raise ValueError("Time series too short for given input_size and forecast_size.")

    for i in range(n):
        X.append(series[i : i + input_size])
        Y.append(series[i + input_size : i + input_size + forecast_size])

    return torch.stack(X), torch.stack(Y)


def train_test_split(X, Y, train_frac=0.8):
    n = X.shape[0]
    n_train = int(n * train_frac)
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


# -------------------------------------------------
# Training utilities (still no classes)
# -------------------------------------------------

def make_optimizer(params, lr=1e-3, weight_decay=0.0):
    return torch.optim.Adam(list(params.values()), lr=lr, weight_decay=weight_decay)


def mse(a, b):
    return ((a - b) ** 2).mean()


def block_loss(x, y, params, layers=4, lambda_backcast=1.0, lambda_forecast=1.0):
    backcast, forecast = nbeats_block_forward(x, params, layers=layers)
    loss_f = mse(forecast, y)
    loss_b = mse(backcast, x)
    return lambda_forecast * loss_f + lambda_backcast * loss_b


def iterate_minibatches(X, Y, batch_size=32, shuffle=True):
    n = X.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        j = idx[i : i + batch_size]
        yield X[j], Y[j]


def _to_leaf_param(tensor, device):
    """
    Make a leaf tensor on device suitable for optimizers.
    (Avoids: ValueError: can't optimize a non-leaf Tensor)
    """
    return tensor.detach().to(device).requires_grad_(True)


def train(
    X_train,
    Y_train,
    X_val,
    Y_val,
    params,
    layers=4,
    lr=1e-3,
    weight_decay=0.0,
    epochs=200,
    batch_size=32,
    lambda_backcast=1.0,
    lambda_forecast=1.0,
    clip_grad_norm=1.0,
    device="cpu",
):
    # move params to device AND keep them as leaf tensors
    for k in list(params.keys()):
        params[k] = _to_leaf_param(params[k], device=device)

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    opt = make_optimizer(params, lr=lr, weight_decay=weight_decay)

    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        train_losses = []
        for xb, yb in iterate_minibatches(X_train, Y_train, batch_size=batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            loss = block_loss(
                xb, yb, params, layers=layers,
                lambda_backcast=lambda_backcast,
                lambda_forecast=lambda_forecast,
            )
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(list(params.values()), clip_grad_norm)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            val_loss = float(
                block_loss(
                    X_val, Y_val, params, layers=layers,
                    lambda_backcast=lambda_backcast,
                    lambda_forecast=lambda_forecast,
                ).detach().cpu()
            )

        tr = sum(train_losses) / max(1, len(train_losses))
        history["train"].append(tr)
        history["val"].append(val_loss)

        if ep % 20 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:4d} | train {tr:.6f} | val {val_loss:.6f}")

    return params, history


#@torch.no_grad()
def predict_forecast(X, params, layers=4, device="cpu"):
    X = X.to(device)
    _, forecast = nbeats_block_forward(X, params, layers=layers)
    return forecast.detach().cpu()


# -------------------------------------------------
# End-to-end runnable section
# -------------------------------------------------

def run_passenger_example(
    csv_path="passenger.csv",
    value_col="Passengers",   # <-- FIX: match typical column name
    input_size=72,
    forecast_size=12,
    layers=4,
    layer_size=512,
    train_frac=0.8,
    lr=1e-3,
    epochs=300,
    batch_size=16,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    series, mean, std = load_passenger_series(csv_path=csv_path, value_col=value_col)
    X, Y = make_windows(series, input_size=input_size, forecast_size=forecast_size)
    X_train, Y_train, X_val, Y_val = train_test_split(X, Y, train_frac=train_frac)

    params = init_block(
        input_size=input_size,
        backcast_size=input_size,
        forecast_size=forecast_size,
        layers=layers,
        layer_size=layer_size,
    )

    params, history = train(
        X_train, Y_train, X_val, Y_val,
        params=params,
        layers=layers,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        lambda_backcast=0.2,
        lambda_forecast=1.0,
        device=device,
    )

    yhat_val = predict_forecast(X_val, params, layers=layers, device=device)

    yhat_val_denorm = yhat_val * std + mean
    y_val_denorm = Y_val * std + mean

    return {
        "params": params,
        "history": history,
        "yhat_val": yhat_val_denorm,
        "y_val": y_val_denorm,
        "mean": mean,
        "std": std,
        "device": device,
    }


###########################
# Run
############################

results = run_passenger_example(
    csv_path="~/adrive/data/ts/passenger.csv",
    value_col="passengers",   # <-- ensure correct column name
    input_size=36,
    forecast_size=12,
    epochs=200,
    batch_size=16,
)

print("\nTraining finished.")
print("Validation forecast shape:", results["yhat_val"].shape)
print("Validation target shape:", results["y_val"].shape)

forecast = results["yhat_val"]

last_forecast = forecast[-1].numpy()


test_y = results["y_val"]  


df_compare = pd.DataFrame({
    "y": test_y[-1].numpy(),
    "fct": last_forecast
})

df_compare["step"] = range(1, len(df_compare) + 1)


from plotnine import * 
(
ggplot(df_compare, aes(x="y", y="fct")) + geom_point() 
    + geom_abline(slope=1, intercept=0, color="red") 
    + theme_minimal()
)

(
    ggplot(df_compare)
    + geom_line(aes(x='step', y='y', color='"Actual"'))
    + geom_line(aes(x='step', y='fct', color='"Forecast"'))
    + scale_color_manual(values={
        "Actual": "blue",
        "Forecast": "orange"
    })
    + labs(color="Series")
    + theme_minimal()
)
