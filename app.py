import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Transformer Visualizer", layout="wide")

# --- HELPER FUNCTIONS (NUMPY TRANSFORMER MATH) ---
def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Transformer Parameters")
input_text = st.sidebar.text_input("Input Text", "Transformers are amazing models")
d_model = st.sidebar.slider("Embedding Dimension (d_model)", min_value=16, max_value=128, value=32, step=16)
num_heads = st.sidebar.slider("Number of Attention Heads", min_value=1, max_value=8, value=4, step=1)
num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=6, value=1, step=1)

show_pe = st.sidebar.toggle("Show Positional Encoding Details", value=True)
show_attn = st.sidebar.toggle("Show Attention Maps Details", value=True)

# Validate heads/dim
if d_model % num_heads != 0:
    st.sidebar.error("Embedding Dimension must be divisible by Number of Heads!")
    st.stop()

d_k = d_model // num_heads

st.title("🧩 Interactive Transformer Visualizer")
st.markdown("Explore how a sequence is processed through a Transformer architecture. *Note: Weights are randomly initialized to demonstrate data flow and tensor shapes.*")

# --- SIMULATED PIPELINE ---
# 1. Tokenization (Simple whitespace split for demo)
tokens = input_text.split()
seq_len = len(tokens)

if seq_len == 0:
    st.warning("Please enter some text in the sidebar.")
    st.stop()

# Create dummy vocabulary mapping
vocab = {word: i for i, word in enumerate(set(tokens))}
token_ids = [vocab[word] for word in tokens]

# 2. Embeddings
np.random.seed(42) # For reproducible random visuals
embedding_matrix = np.random.randn(len(vocab), d_model)
embeddings = np.array([embedding_matrix[tid] for tid in token_ids])

# 3. Positional Encoding
pe = get_positional_encoding(seq_len, d_model)
x = embeddings + pe

# --- UI TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Tokenization", "2. Positional Encoding", "3. Self-Attention", 
    "4. Multi-Head", "5. FFN", "6. Residual & Norm", "7. Layer Stacking"
])

with tab1:
    st.header("1. Tokenization & Input Representation")
    st.markdown("**Concept:** Text is split into tokens and mapped to numerical vectors (embeddings).")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Tokens & IDs**")
        df_tokens = pd.DataFrame({"Token": tokens, "ID": token_ids})
        st.dataframe(df_tokens, use_container_width=True)
    with col2:
        st.write("**Embedding Matrix (Subset)**")
        fig = px.imshow(embeddings, labels=dict(x="Dimension", y="Token"), y=tokens, aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("2. Positional Encoding")
    st.markdown("**Concept:** Adding wave-like patterns so the model understands the order of words.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Positional Encodings (PE)**")
        fig_pe = px.imshow(pe, labels=dict(x="Dimension", y="Position"), y=tokens, aspect="auto", color_continuous_scale="RdBu")
        st.plotly_chart(fig_pe, use_container_width=True)
    with col2:
        st.write("**Embeddings + PE**")
        fig_x = px.imshow(x, labels=dict(x="Dimension", y="Position"), y=tokens, aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig_x, use_container_width=True)

with tab3:
    st.header("3. Self-Attention Mechanism")
    st.markdown("**Concept:** Words look at other words to gather context. We compute Queries (Q), Keys (K), and Values (V).")
    st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
    
    # Simulate W_q, W_k, W_v for a single head demo
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    
    scores = (Q @ K.T) / np.sqrt(d_model)
    attention_weights = softmax(scores)
    attention_output = attention_weights @ V
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Q Matrix**")
        st.plotly_chart(px.imshow(Q, y=tokens, aspect="auto"), use_container_width=True)
    with col2:
        st.write("**K Matrix**")
        st.plotly_chart(px.imshow(K, y=tokens, aspect="auto"), use_container_width=True)
    with col3:
        st.write("**Attention Weights (Heatmap)**")
        st.plotly_chart(px.imshow(attention_weights, x=tokens, y=tokens, aspect="auto", color_continuous_scale="Blues"), use_container_width=True)

with tab4:
    st.header("4. Multi-Head Attention")
    st.markdown("**Concept:** The embedding dimension is split into multiple heads, allowing the model to focus on different aspects of context simultaneously.")
    
    st.write(f"Splitting `d_model` ({d_model}) into `{num_heads}` heads of size `{d_k}`.")
    
    cols = st.columns(num_heads)
    concat_output = []
    
    for i in range(num_heads):
        # Slice Q, K, V for each head
        Q_h = Q[:, i*d_k : (i+1)*d_k]
        K_h = K[:, i*d_k : (i+1)*d_k]
        V_h = V[:, i*d_k : (i+1)*d_k]
        
        scores_h = (Q_h @ K_h.T) / np.sqrt(d_k)
        attn_weights_h = softmax(scores_h)
        head_out = attn_weights_h @ V_h
        concat_output.append(head_out)
        
        with cols[i]:
            st.write(f"**Head {i+1} Weights**")
            st.plotly_chart(px.imshow(attn_weights_h, x=tokens, y=tokens, aspect="auto", color_continuous_scale="Reds"), use_container_width=True)
            
    multi_head_out = np.concatenate(concat_output, axis=-1)

with tab5:
    st.header("5. Feedforward Network (FFN)")
    st.markdown("**Concept:** A dense neural network applied to each token independently to add non-linearity.")
    
    # Simulate FFN: Linear -> ReLU -> Linear
    d_ff = d_model * 4
    W1 = np.random.randn(d_model, d_ff)
    W2 = np.random.randn(d_ff, d_model)
    
    ffn_hidden = np.maximum(0, multi_head_out @ W1) # ReLU
    ffn_out = ffn_hidden @ W2
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Input to FFN**")
        st.plotly_chart(px.imshow(multi_head_out, y=tokens, aspect="auto"), use_container_width=True)
    with col2:
        st.write("**Output after FFN**")
        st.plotly_chart(px.imshow(ffn_out, y=tokens, aspect="auto"), use_container_width=True)

with tab6:
    st.header("6. Residual Connections & Normalization")
    st.markdown("**Concept:** Adding the input back to the output (Residual) and normalizing to prevent exploding gradients.")
    
    residual = x + multi_head_out
    normed = layer_norm(residual)
    
    # Compare variances to show normalization effect
    var_before = np.var(residual, axis=-1)
    var_after = np.var(normed, axis=-1)
    
    df_var = pd.DataFrame({"Token": tokens, "Variance Before Norm": var_before, "Variance After Norm": var_after})
    
    st.write("**Effect of Layer Normalization on Token Variance**")
    fig_norm = go.Figure()
    fig_norm.add_trace(go.Bar(x=df_var["Token"], y=df_var["Variance Before Norm"], name="Before Norm"))
    fig_norm.add_trace(go.Bar(x=df_var["Token"], y=df_var["Variance After Norm"], name="After Norm"))
    fig_norm.update_layout(barmode='group')
    st.plotly_chart(fig_norm, use_container_width=True)
    
    st.write("**Normalized Output State**")
    st.plotly_chart(px.imshow(normed, y=tokens, aspect="auto", color_continuous_scale="Viridis"), use_container_width=True)

with tab7:
    st.header("7. Layer-wise Representation Evolution")
    st.markdown("**Concept:** As tokens pass through multiple layers, their representations become increasingly context-aware.")
    
    # Simulate passing through multiple layers
    current_state = x
    layer_states = [current_state]
    
    for l in range(num_layers):
        # Simulated attention and FFN step for layer l
        Q_l = current_state @ np.random.randn(d_model, d_model)
        K_l = current_state @ np.random.randn(d_model, d_model)
        V_l = current_state @ np.random.randn(d_model, d_model)
        
        attn_out = softmax((Q_l @ K_l.T) / np.sqrt(d_model)) @ V_l
        current_state = layer_norm(current_state + attn_out) # Add & Norm
        
        ffn_out = np.maximum(0, current_state @ np.random.randn(d_model, d_model*4)) @ np.random.randn(d_model*4, d_model)
        current_state = layer_norm(current_state + ffn_out) # Add & Norm
        
        layer_states.append(current_state)
        
    st.write(f"Tracking representations across **{num_layers} layers**:")
    
    cols = st.columns(len(layer_states))
    for i, state in enumerate(layer_states):
        with cols[i]:
            title = "Input (Layer 0)" if i == 0 else f"Output Layer {i}"
            st.write(f"**{title}**")
            st.plotly_chart(px.imshow(state, y=tokens, aspect="auto"), use_container_width=True)