function z = dejong1fcn(xx)

    % De Jong's function n.1 (Sphere)

    % Search domain: [-5.12,5.12]
    % Global minimum: f(x) = 0 | x = (0,...,0)

    % xx = max(-5.12,min(5.12,xx));

    z = sum(xx.^2,2);

end