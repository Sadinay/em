function z = griewankfcn(xx)

    % Griewank function

    % Search domain: [-600,600]
    % Global minimum: f(x) = 0 | x = (0,...,0)

    % xx = max(-600,min(600,xx));

    d = size(xx,2);

    z = sum(xx.^2,2)/4000 - prod(cos(xx/sqrt(d)),2) + 1;

end