function z = rosenbrockfcn(xx)

    % Rosenbrock's function

    % Search domain: [-5,10]
    % Global minimum: f(x) = 0 | x = (1,...,1)

    % xx = max(-5,min(10,xx));

    d = size(xx,2);

    z = sum(100*(xx(:,d) - xx(:,d-1).^2).^2 + (xx(:,d-1) - 1).^2,2);

end