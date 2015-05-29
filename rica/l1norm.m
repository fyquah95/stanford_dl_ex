function Z = l1norm(x, epsilon)

    Z = sum(sum(sqrt(x .^ 2 + epsilon)));