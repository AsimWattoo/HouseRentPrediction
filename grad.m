function g = grad(theta, X, y, alpha, lambda)
    m = length(y);
    g = theta - (alpha * (1/ m)) * ((theta' * X' - y') * X)' + (lambda / m) * theta;
end