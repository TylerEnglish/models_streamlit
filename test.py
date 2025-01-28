import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

import plotly.graph_objects as go

from captum.attr import IntegratedGradients
import shap
from lime.lime_tabular import LimeTabularExplainer

###############################################################################
# 1) DATA GENERATION & PREPARATION
###############################################################################
N_SAMPLES = 2000
N_FEATURES = 10
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.5, 0.5],
    random_state=42
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert to torch Tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)
X_test_t  = torch.FloatTensor(X_test_scaled)
y_test_t  = torch.LongTensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)
BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

###############################################################################
# 2) DEFINE & TRAIN A SIMPLE MLP
###############################################################################
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleMLP(input_dim=N_FEATURES, hidden_dim=32, output_dim=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            out = model(bx)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
    acc = accuracy_score(y_test, preds)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}, Test Acc={acc*100:.2f}%")

###############################################################################
# 3) MODEL PERFORMANCE EVALUATION
###############################################################################
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, preds)
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_test, preds))

###############################################################################
# 4) GLOBAL INTERPRETATION: PERMUTATION IMPORTANCE
###############################################################################
class PyTorchWrapper(BaseEstimator):
    def __init__(self, net):
        self.net = net
        self.is_fitted_ = True
    def fit(self, X, y):
        return self
    def predict(self, X):
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            logits = self.net(X_t)
            return torch.argmax(logits, dim=1).cpu().numpy()
    def predict_proba(self, X):
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            logits = self.net(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs
    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

wrapper = PyTorchWrapper(model)

perm_results = permutation_importance(
    estimator=wrapper,
    X=X_test_scaled,
    y=y_test,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

feature_names = [f"Feature_{i}" for i in range(N_FEATURES)]
means = perm_results.importances_mean
stds  = perm_results.importances_std
sorted_idx = np.argsort(means)[::-1]

print("\n--- Permutation Importance (Global) ---")
for i in sorted_idx:
    print(f"{feature_names[i]} => Mean={means[i]:.4f}, Std={stds[i]:.4f}")

###############################################################################
# 5) CAPTUM INTEGRATED GRADIENTS (Local)
###############################################################################
def captum_forward_func(x):
    """
    We do NOT wrap in torch.no_grad() so that
    gradients can flow w.r.t. input for IG.
    """
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    return probs[:,1]  # Probability of class=1

ig = IntegratedGradients(captum_forward_func)

num_explanations = 2
sample_idxs = np.random.choice(len(X_test_scaled), num_explanations, replace=False)

print("\n--- Captum Integrated Gradients ---")
for idx in sample_idxs:
    # Mark input sample as requiring grad
    inp = torch.FloatTensor(X_test_scaled[idx:idx+1]).requires_grad_()
    baseline = torch.zeros_like(inp)
    
    attributions = ig.attribute(inp, baselines=baseline, n_steps=50)
    local_attr = attributions[0].detach().cpu().numpy()
    
    sample_prob = wrapper.predict_proba(X_test_scaled[idx:idx+1])[0,1]
    pred_label  = 1 if sample_prob>=0.5 else 0
    true_label  = y_test[idx]
    print(f"\nTest Sample {idx}: True={true_label}, P(1)={sample_prob:.3f}, Pred={pred_label}")
    # Sort by magnitude
    pairs = list(zip(feature_names, local_attr))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(" Top attributions:")
    for feat, val in pairs[:5]:
        print(f"  {feat}: {val:.4f}")

###############################################################################
# 6) SHAP (Kernel Explainer) - Handle Single or Double Output
###############################################################################
def shap_predict_fn(data_np):
    with torch.no_grad():
        xt = torch.FloatTensor(data_np)
        logits = model(xt)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

explainer_shap = shap.KernelExplainer(shap_predict_fn, X_train_scaled[:50])

rand_idx = np.random.choice(len(X_test_scaled), 1, replace=False)[0]
test_row = X_test_scaled[rand_idx:rand_idx+1]
shap_vals = explainer_shap.shap_values(test_row, nsamples=100)

print(f"\n--- SHAP Explanation for sample idx={rand_idx} ---")
if isinstance(shap_vals, list):
    print(f"Detected shap_values is a list of length {len(shap_vals)} (one array per class).")
    for c, sv in enumerate(shap_vals):
        print(f" Class {c} shap => {sv}")
else:
    print("Detected shap_values is a single array =>", shap_vals)

###############################################################################
# 7) LIME (Local Surrogate)
###############################################################################
lime_explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    class_names=["Class0", "Class1"],
    discretize_continuous=True
)

def lime_predict_fn(data_np):
    model.eval()
    with torch.no_grad():
        xt = torch.FloatTensor(data_np)
        logits = model(xt)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

lime_exp = lime_explainer.explain_instance(
    data_row=test_row[0],
    predict_fn=lime_predict_fn,
    num_features=5
)
print("\n--- LIME Explanation (Console) ---")
print(lime_exp.as_list())

###############################################################################
# 8) 2D COUNTERFACTUAL SHIFT (Flipping Label) via Direct Logit Grad
###############################################################################
pca = PCA(n_components=2)
pca.fit(X_train_scaled)

X_train_2d = pca.transform(X_train_scaled)
X_test_2d  = pca.transform(X_test_scaled)

def pca_inverse_predict_proba_1(x_2d):
    """
    Convert 2D PCA point => original 10D => run model => prob(class=1).
    """
    x_10d = pca.inverse_transform(x_2d.reshape(1,-1))
    return wrapper.predict_proba(x_10d)[0,1]

def iterative_logit_shift_to_class(
    x_10d,
    target_class=1,
    target_prob=0.5,
    step=0.05,
    max_steps=50,
    feature_range=None
):
    if feature_range is None:
        min_arr = np.full_like(x_10d, -1e9)
        max_arr = np.full_like(x_10d,  1e9)
    else:
        min_arr, max_arr = feature_range

    original = x_10d.copy()
    x_mod    = x_10d.copy()

    def done(prob):
        if target_class == 1:
            return prob >= target_prob
        else:
            return prob <= (1.0 - target_prob)

    # Evaluate initial prob
    prob = wrapper.predict_proba(x_mod.reshape(1,-1))[0,1]
    steps_used = 0

    while (not done(prob)) and (steps_used < max_steps):
        # 1) We'll compute gradient of logit(1) w.r.t. x_mod
        x_tensor = torch.FloatTensor(x_mod).unsqueeze(0)
        x_tensor.requires_grad_()

        # Forward pass
        logits = model(x_tensor)
        logit_1 = logits[:,1]  # scalar for class=1 logit

        # If we want target_class=0 => we do negative gradient to reduce logit(1)
        if target_class == 0:
            loss = logit_1  # we'll do a gradient descent step on this
        else:
            loss = -logit_1

        # Compute gradient
        grad_list = torch.autograd.grad(loss, x_tensor)
        grad = grad_list[0].detach().cpu().numpy()[0]  # shape [10]

        # 2) Update x_mod
        if target_class == 0:
            x_mod = x_mod - step * grad
        else:
            x_mod = x_mod - step * grad  # because loss = -logit => grad= negative
            # So we do a negative step to go in the direction that *increases* logit(1)

        # clamp
        x_mod = np.clip(x_mod, min_arr, max_arr)

        # check new prob
        prob = wrapper.predict_proba(x_mod.reshape(1,-1))[0,1]
        steps_used += 1

    # Record changes
    changed = {}
    for i in range(len(original)):
        if abs(x_mod[i] - original[i]) > 1e-7:
            changed[i] = (original[i], x_mod[i])

    return x_mod, prob, steps_used, changed

# Example usage:
orig_10d = test_row[0]
orig_prob = wrapper.predict_proba(orig_10d.reshape(1,-1))[0,1]

# SHIFT TOWARD class=0 or class=1
desired_class = 0
desired_thresh = 0.5

x_shifted, final_prob, used_steps, changed_feats = iterative_logit_shift_to_class(
    x_10d=orig_10d,
    target_class=desired_class,
    target_prob=desired_thresh,
    step=0.05,
    max_steps=50
)

# Visualize in 2D
orig_2d    = pca.transform(orig_10d.reshape(1,-1))[0]
shifted_2d = pca.transform(x_shifted.reshape(1,-1))[0]

plt.figure(figsize=(8,6))
cls_name = "Class=0" if desired_class==0 else "Class=1"
plt.title(f"Direct Logit Shift => {cls_name}, threshold={desired_thresh}")

# Plot distribution
plt.scatter(X_train_2d[y_train==0,0], X_train_2d[y_train==0,1],
            color='white', edgecolor='black', s=40, alpha=0.7, label='Class0')
plt.scatter(X_train_2d[y_train==1,0], X_train_2d[y_train==1,1],
            color='black', s=40, alpha=0.7, label='Class1')

# Probability contours
xx = np.linspace(X_train_2d[:,0].min()-1, X_train_2d[:,0].max()+1, 100)
yy = np.linspace(X_train_2d[:,1].min()-1, X_train_2d[:,1].max()+1, 100)
XX,YY = np.meshgrid(xx, yy)
grid_2d = np.c_[XX.ravel(), YY.ravel()]
probs_grid = np.array([pca_inverse_predict_proba_1(pt) for pt in grid_2d])
probs_grid = probs_grid.reshape(XX.shape)
plt.contourf(XX, YY, probs_grid, levels=15, cmap='bone', alpha=0.6)

# Mark original & shifted
plt.scatter(orig_2d[0],    orig_2d[1], color='red',   s=100, label='Original', zorder=5)
plt.scatter(shifted_2d[0], shifted_2d[1], color='green', s=100, label='Shifted', zorder=5)
plt.arrow(orig_2d[0], orig_2d[1],
          shifted_2d[0]-orig_2d[0], shifted_2d[1]-orig_2d[1],
          head_width=0.05, length_includes_head=True, color='red', zorder=6)

plt.legend()
plt.show()

old_label = int(orig_prob >= 0.5)
new_prob  = final_prob
new_label = int(new_prob >= 0.5)

print("\n=== DIRECT LOGIT SHIFT RESULTS ===")
print(f"Desired flipping to: Class={desired_class}")
print(f"Original P(class=1)={orig_prob:.3f}, label={old_label}")
print(f"Shifted  P(class=1)={new_prob:.3f}, label={new_label}")
print(f"Steps used={used_steps}")

old_label = 1 if orig_prob >= 0.5 else 0
new_label = 1 if final_prob >= 0.5 else 0
print(f"Old label = {old_label}, New label = {new_label} (did we flip it?)")
if changed_feats:
    print("Feature changes:")
    for i, (oldv, newv) in changed_feats.items():
        direction = "↑" if newv>oldv else "↓"
        print(f"  Feature_{i}: {oldv:.3f} -> {newv:.3f}  ({direction} by {abs(newv - oldv):.3f})")
else:
    print("No features changed (already in desired region or model is locked).")

###############################################################################
# 9) VISUALIZING LAYER WEIGHTS & BIASES
###############################################################################
layers = [model.fc1, model.fc2, model.fc3]
layer_names = ["Layer1 (fc1)", "Layer2 (fc2)", "Output (fc3)"]

avg_abs_weights = []
bias_values     = []

for i, layer in enumerate(layers):
    w = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy()
    avg_abs = np.mean(np.abs(w))
    avg_abs_weights.append(avg_abs)
    bias_values.append(b)
    
    print(f"\n--- {layer_names[i]} ---")
    print(f" Weight shape = {w.shape}, avg|weight|= {avg_abs:.4f}")
    print(f" Bias shape   = {b.shape}, sample= {b[:5]}...")

plt.figure(figsize=(6,4))
plt.bar(layer_names, avg_abs_weights, color=['blue','green','orange'])
plt.title("Average Absolute Weight by Layer")
plt.ylabel("Avg |Weight|")
plt.xticks(rotation=20)
plt.grid(True, axis='y')
plt.show()


###############################################################################
# 10) INTERACTIVE FLOW THROUGH THE NETWORK via PLOTLY SANKEY (COOLER VERSION)
###############################################################################
def visualize_forward_pass_sankey(model, sample_np, top_k=5):
    # 1) Convert the input sample to float array
    x_np = np.array(sample_np, dtype=np.float32)  # shape [N_FEATURES]

    # 2) Extract weights & biases
    with torch.no_grad():
        w1 = model.fc1.weight.detach().cpu().numpy()  # [32, 10]
        b1 = model.fc1.bias.detach().cpu().numpy()    # [32]
        w2 = model.fc2.weight.detach().cpu().numpy()  # [32, 32]
        b2 = model.fc2.bias.detach().cpu().numpy()    # [32]
        w3 = model.fc3.weight.detach().cpu().numpy()  # [2, 32]
        b3 = model.fc3.bias.detach().cpu().numpy()    # [2]

    # 3) Manual forward pass
    hidden1_linear     = w1 @ x_np + b1   # shape [32]
    hidden1_activated  = np.maximum(0, hidden1_linear)
    hidden2_linear     = w2 @ hidden1_activated + b2
    hidden2_activated  = np.maximum(0, hidden2_linear)
    logits             = w3 @ hidden2_activated + b3
    exps               = np.exp(logits - np.max(logits))
    probs              = exps / exps.sum()

    # Select top_k neurons by absolute activation in hidden1 & hidden2
    h1_abs = np.abs(hidden1_activated)
    h1_top_idx = np.argsort(h1_abs)[::-1][:top_k]
    h1_info = [{"index": i, 
                "pre": hidden1_linear[i], 
                "post": hidden1_activated[i]} 
                for i in h1_top_idx]

    h2_abs = np.abs(hidden2_activated)
    h2_top_idx = np.argsort(h2_abs)[::-1][:top_k]
    h2_info = [{"index": i, 
                "pre": hidden2_linear[i], 
                "post": hidden2_activated[i]} 
                for i in h2_top_idx]

    out_info = []
    for c in range(2):
        out_info.append({
            "index": c,
            "logit": logits[c],
            "prob": probs[c]
        })

    # Prepare Sankey nodes
    input_dim = len(x_np)
    node_labels = []
    node_colors = []

    # Input feature nodes
    for i in range(input_dim):
        node_labels.append(f"Input_{i} (val={x_np[i]:.2f})")
        node_colors.append("sienna")

    # Hidden1
    base_h1 = input_dim
    hidden1_idx_map = {}
    for j, hinfo in enumerate(h1_info):
        idx = hinfo["index"]
        label_str = (f"H1_{idx}<br>"
                     f"pre={hinfo['pre']:.2f}, "
                     f"post={hinfo['post']:.2f}")
        hidden1_idx_map[idx] = base_h1 + j
        node_labels.append(label_str)
        node_colors.append("royalblue")

    # Hidden2
    base_h2 = input_dim + len(h1_info)
    hidden2_idx_map = {}
    for j, hinfo in enumerate(h2_info):
        idx = hinfo["index"]
        label_str = (f"H2_{idx}<br>"
                     f"pre={hinfo['pre']:.2f}, "
                     f"post={hinfo['post']:.2f}")
        hidden2_idx_map[idx] = base_h2 + j
        node_labels.append(label_str)
        node_colors.append("forestgreen")

    # Output
    base_out = input_dim + len(h1_info) + len(h2_info)
    out_idx_map = {}
    out_colors = ["tomato","tomato"]
    for j, outc in enumerate(out_info):
        idx = outc["index"]
        label_str = (f"Out_{idx}<br>"
                     f"logit={outc['logit']:.2f}, "
                     f"p={outc['prob']:.2f}")
        out_idx_map[idx] = base_out + j
        node_labels.append(label_str)
        node_colors.append(out_colors[j])

    # Prepare Sankey links
    link_sources = []
    link_targets = []
    link_values  = []
    link_labels  = []
    link_colors  = []

    def link_color(flow_val):
        # If flow_val >=0 => green, else red
        if flow_val >= 0:
            return "rgba(50,205,50,0.7)"   # bright green
        else:
            return "rgba(220,20,60,0.7)"  # crimson

    # 1) input -> hidden1
    for i in range(input_dim):
        val_i = x_np[i]
        for h1j in h1_info:
            j_idx = h1j["index"]
            w_ij = w1[j_idx, i]
            flow_val = w_ij * val_i
            link_sources.append(i)
            link_targets.append(hidden1_idx_map[j_idx])
            link_values.append(abs(flow_val))
            link_labels.append(f"(input={val_i:.2f}, w={w_ij:.2f})")
            link_colors.append(link_color(flow_val))

    # 2) hidden1 -> hidden2
    for h1j in h1_info:
        j_idx  = h1j["index"]
        post_j = h1j["post"]
        for h2k in h2_info:
            k_idx   = h2k["index"]
            w_jk    = w2[k_idx, j_idx]
            flow_val= w_jk * post_j
            link_sources.append(hidden1_idx_map[j_idx])
            link_targets.append(hidden2_idx_map[k_idx])
            link_values.append(abs(flow_val))
            link_labels.append(f"(act={post_j:.2f}, w={w_jk:.2f})")
            link_colors.append(link_color(flow_val))

    # 3) hidden2 -> output
    for h2k in h2_info:
        k_idx  = h2k["index"]
        post_k = h2k["post"]
        for outc in out_info:
            c_idx   = outc["index"]
            w_kc    = w3[c_idx, k_idx]
            flow_val= w_kc * post_k
            link_sources.append(hidden2_idx_map[k_idx])
            link_targets.append(out_idx_map[c_idx])
            link_values.append(abs(flow_val))
            link_labels.append(f"(act={post_k:.2f}, w={w_kc:.2f})")
            link_colors.append(link_color(flow_val))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=1.0),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            label=link_labels,
            color=link_colors,
            hovertemplate="<b>Flow:</b> %{value}<br />%{label}<extra></extra>"
        )
    )])

    # Detailed legend in the layout
    fig.update_layout(
        title_text=(f"<b>Sankey Flow: Input → Hidden1 → Hidden2 → Output</b> (top_k={top_k})<br>"
                    "Green = Positive push → Nudges outcome toward class=1 | Red = Negative push → Nudges outcome toward class=0"),
        font=dict(size=13),
        annotations=[
            dict(
                x=1, y=-0.12, xref='paper', yref='paper', showarrow=False,
                text=("Flow thickness = <b>|weight × activation|</b>.<br>"
                      "Positive (green) => pushing next neuron’s sum higher => more 'pro-1'<br>"
                      "Negative (red) => pushing next neuron’s sum lower => more 'pro-0'")
            )
        ]
    )

    fig.show()

    pred_class = int(np.argmax(probs))
    print(f"Final Prediction => Class {pred_class}, Probability = {probs[pred_class]:.4f}")

def test_input_flow(model, top_k=5):
    """
    Lets you input a 10-dimensional vector from the console 
    (e.g. "1.0 -0.5 0.3 ..." ) to see how the Sankey changes.
    """
    user_input = input("Enter 10 feature values (space-separated), e.g.: 1.0 -0.5 0.2 ...\n").strip()
    values = [float(x) for x in user_input.split()]
    if len(values) != 10:
        print("Please provide exactly 10 values!")
        return
    visualize_forward_pass_sankey(model, values, top_k=top_k)

test_input_flow(model, top_k=5)
