w1 = 784, 128
w2 = 128, 64
w3 = 64, 10

z, a (1) = 32, 128
z, a (2) = 32, 64
z, a (3) = 32, 10

-   cap nhat w3:

    -   err = dao lam loss \* dao ham active (z3)
        -   32, 10 \* 32, 10 => 32, 10
    -   param(a2).T @ err
        -   (32, 64).T @ (32, 10) => 64, 10

-   cap nhat w2:

    -   err = err @ w3 \* dao ham active (z2)
        -   32, 10 @ (64, 10).T \* 32, 64 => 32, 64
    -   param(a1).T @ err
        -   (32, 128).T @ 32, 64 = 128, 64

-   cap nhat w1:
    -   err = err @ w2 \* dao ham active (z1)
        -   32, 64 @ (128, 64).T \* 32, 128 => 32, 128
    -   param(x).T@ err
        -   (32, 784).T @ (32, 128) => 784, 128
