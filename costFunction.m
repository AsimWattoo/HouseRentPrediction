function J = costFunction(theta, X, y, lambda)
    m = length(y);
    J = (1 / (2 * m)) * sum((predict(theta, X) - y').^2) + (lambda / (2 * m)) * sum(theta .^2);
end