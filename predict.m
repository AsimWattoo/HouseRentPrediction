function [p] = predict(theta, X)
    p = theta' * X';
end