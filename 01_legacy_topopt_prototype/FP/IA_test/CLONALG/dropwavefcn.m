function z = dropwavefcn(xx)

    % Drop-Wave function

    % Search domain: [-5.12,5.12]
    % Global minimum: f(x) = -1 | x = (0,0)

    % xx = max(-5.12,min(5.12,xx));

    x1 = xx(:,1);
    x2 = xx(:,2);


    frac1 = 1 + cos(12*sqrt(x1.^2 + x2.^2));
    frac2 = 0.5*(x1.^2 + x2.^2) + 2;

    z = - frac1./frac2;

end