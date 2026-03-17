import numpy as np


def safe_divide(a, b, eps=1e-8):
    return a / (b + eps)


def compute_physics_features(df):
    out = df.copy()

    out["ptj1"] = np.sqrt(out["pxj1"]**2 + out["pyj1"]**2)
    out["ptj2"] = np.sqrt(out["pxj2"]**2 + out["pyj2"]**2)

    out["Ej1"] = np.sqrt(out["pxj1"]**2 + out["pyj1"]**2 + out["pzj1"]**2 + out["mj1"]**2)
    out["Ej2"] = np.sqrt(out["pxj2"]**2 + out["pyj2"]**2 + out["pzj2"]**2 + out["mj2"]**2)

    out["pxjj"] = out["pxj1"] + out["pxj2"]
    out["pyjj"] = out["pyj1"] + out["pyj2"]
    out["pzjj"] = out["pzj1"] + out["pzj2"]
    out["Ejj"] = out["Ej1"] + out["Ej2"]

    mjj2 = out["Ejj"]**2 - out["pxjj"]**2 - out["pyjj"]**2 - out["pzjj"]**2
    mjj2 = np.maximum(mjj2, 0.0)
    out["mjj"] = np.sqrt(mjj2)

    out["tau21_j1"] = safe_divide(out["tau2j1"], out["tau1j1"])
    out["tau32_j1"] = safe_divide(out["tau3j1"], out["tau2j1"])
    out["tau21_j2"] = safe_divide(out["tau2j2"], out["tau1j2"])
    out["tau32_j2"] = safe_divide(out["tau3j2"], out["tau2j2"])

    out["pt_balance"] = np.abs(out["ptj1"] - out["ptj2"]) / (out["ptj1"] + out["ptj2"] + 1e-8)
    out["m_ratio"] = out["mj1"] / (out["mj2"] + 1e-8)

    return out
