function z = schwefelfcn(xx)

    % Schwefel's function

    % Search domain: [-500,500]
    % Global minimum: f(x) = 0 | x = (420.9687,...,420.9687)

    % xx = max(-500,min(500,xx));

    d = size(xx,2);

    z = 418.9829 * d - (sum(xx.*sin(sqrt(abs(xx))),2));

end