function z = crossintrayfcn(xx)

    % Cross-in-Tray Function

    % Search domain: [-10,10]
    % Global minimum: f(x) = -2.0626 | x = (+-1.3491)

    % xx = max(-10,min(10,xx));

    x1 = xx(:,1);
    x2 = xx(:,2);

    z = -0.0001*(abs(sin(x1).*sin(x2).*exp(abs(100 - sqrt(x1.^2 + x2.^2)/pi)))+1).^0.1;

end