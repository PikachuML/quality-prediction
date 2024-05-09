# quality-prediction
An end-to-end regression prediction model for variable-length data, which also proposes an effective fusion method for interpretable different structural features.

model pseudocode：

CLASS Blendmapping EXTENDS Module
    INIT(d_model, d_yc, d_y, N, heads, m, dropout)
        input <- LinearLayer(m, d_model)
        encoder <- MatEncoder(d_model, N, heads, dropout)
        Linear1 <- LinearLayer(d_yc, d_model)
        output <- LinearLayer(d_model, d_y)
        dropout <- Dropout(dropout)

    FUNCTION forward(src, prop, yc)
        out <- RELU(input(src))
        out <- encoder(out)
        out <- elementwise multiplication of out and expanded prop along the third dimension
        out <- sum out across dimension 1
        x <- SIGMOID(Linear1(yc))
        out <- elementwise multiplication of out and x
        out <- apply dropout to out
        out <- output(out)
        RETURN out

CLASS MatEncoder EXTENDS Module
    INIT(d_model, N, heads, dropout)
        layers <- get_clones(MatEncoderLayer(d_model, heads, dropout), N)

    FUNCTION forward(x)
        FOR each layer in layers
            x <- layer(x)
        RETURN x

CLASS MatEncoderLayer EXTENDS Module
    INIT(d_model, heads, dropout)
        norm_1 <- Norm(d_model)
        norm_2 <- Norm(d_model)
        attn <- Matricatt(heads, d_model, dropout)
        dropout_1 <- Dropout(dropout)

    FUNCTION forward(x)
        x <- norm_1(x)
        x <- x + attn(x, x, x)
        x <- norm_2(x)
        x <- dropout_1(x)
        RETURN x

CLASS Matricatt EXTENDS Module
    INIT(heads, d_model, dropout)
        d_k <- d_model // heads
        h <- heads
        v_linear <- LinearLayer(d_model, d_model)
        convkq <- Conv2DLayer(1, d_k, 1)
        dropout <- Dropout(dropout)
        out <- LinearLayer(d_model, d_model)

    FUNCTION forward(q, k, v)
        bs <- batch size of q
        v <- reshape v_linear(v) to (bs, -1, h, d_k)
        k <- add dimension to k
        ckq <- apply convkq to k
        ckq <- sum ckq over dimension 2 and reshape to (bs, -1, h, d_k)
        ckq <- transpose ckq dimensions 1 and 2
        v <- transpose v dimensions 1 and 2
        scores <- mat_attention(ckq, v, d_k, dropout) + v
        scores <- transpose scores dimensions 1 and 2 and reshape to (bs, -1, d_model)
        output <- RELU(out(scores))
        RETURN output

FUNCTION get_clones(module, N)
    RETURN list of N deep copies of module

CLASS Norm EXTENDS Module
    INIT(d_model, eps = 1e-6)
        size <- d_model
        alpha <- trainable parameter initialized to ones(size)
        bias <- trainable parameter initialized to zeros(size)
        eps <- eps

    FUNCTION forward(x)
        norm <- alpha * (x - mean of x over last dimension, keep dimension) / (standard deviation of x over last dimension, keep dimension + eps) + bias
        RETURN norm

FUNCTION mat_attention(qk, v, d_k, dropout=None)
    scores <- transpose qk dimensions -2 and -1 divided by sqrt(d_k)
    IF dropout is not None THEN
        scores <- apply dropout to scores
    scores <- softmax of scores along dimension -2
    output <- matrix multiplication of v and scores
    RETURN output
