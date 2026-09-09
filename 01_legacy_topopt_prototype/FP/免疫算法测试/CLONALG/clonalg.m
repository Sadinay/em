% CLONALG - Clonal Selection Algorithm for Optimization Problems
% For 3D MINIMIZATION Problems

% OUTPUTS:

% x,y,fx	-> fx = f(x,y) is the minimal value of the function
% vfx		-> Best fitness of population through generations

% Reference:
% de Castro, L. N., and Von Zuben, F. J. Learning and Optimization Using
% the Clonal Selection Principle. IEEE Transactions on Evolutionary
% Computation. v. 6(3). 2002. DOI: 10.1109/TEVC.2002.1011539

clear
close
clc

rng('default')

N    = 100;                  % Population size
Ab   = cadeia(N,44);         % Antibody population
gen  = 50;                   % Number of generations
pm   = 0.1;                  % Mutation probability
d    = 0.1;                  % Population to suffer random reshuffle
beta = 0.1;                  % Proportion of clones

% Function to optimization
f = @ackleyfcn;

varMin = -32; % Lower bound
varMax =  32; % Upper bound

fbest = 0; % Global f best (minimum)

x = meshgrid(linspace(varMin, varMax, 61));
y = meshgrid(linspace(varMin, varMax, 61))';

vxp = x;
vyp = y;

vzp = f([x(:),y(:)]);
vzp = reshape(vzp,size(x));

x = decode(Ab(:,1:22),varMin,varMax);
y = decode(Ab(:,23:end),varMin,varMax);

fit = f([x(:),y(:)]);

figure
imprime(1,vxp,vyp,vzp,x,y,fit)

% Hypermutation controlling parameters
pma  = pm;
itpm = gen;
pmr  = 0.8;

% General defintions
vfx   = zeros(gen,1);
PRINT = 1;

% Generations
for it = 1:gen
    
    % Decode (Step 2)
    x = decode(Ab(:,1:22),varMin,varMax);
    y = decode(Ab(:,23:end),varMin,varMax);
    
    fit = f([x(:),y(:)]);
    
    [a,ind] = sort(fit);
    
    % Select (Step 3)
    valx = x(ind);
    valy = y(ind);
    
    imprime(PRINT,vxp,vyp,vzp,x,y,fit);

    % Clone (Step 4)
    [T,pcs] = reprod(N,beta,ind,Ab);

    % Hypermutation (Step 5)
    M = rand(size(T,1),44) <= pm;
    T = T - 2 .* (T.*M) + M;
    
    T(pcs,:) = Ab(fliplr(ind),:);
    
    % Decode (Step 6)
    x = decode(T(:,1:22),varMin,varMax);
    y = decode(T(:,23:end),varMin,varMax);
    
    fit = f([x(:),y(:)]);
    
    pcs = [0 pcs];

    for i = 1:N
        % Re-Selection (Step 7)
        [~,bcs(i)] = min(fit(pcs(i)+1:pcs(i+1)));
        bcs(i) = bcs(i) + pcs(i);
    end

    % Insert
    Ab(fliplr(ind),:) = T(bcs,:);

    % Editing (Repertoire shift)
    nedit = round(d*N);

    % Replace (Step 8)
    Ab(ind(end-(nedit-1):end),:) = cadeia(nedit,44);

    pm = pmcont(pm,pma,pmr,it,itpm);
    
    vfx(it) = a(1);
        
    % fprintf('%2d  f(%6.2f,%6.2f): %7.2f\n',it,valx(1),valy(1),vfx(it))

end

% Minimization problem
x  = valx(1);
y  = valy(1);
fx = vfx(end);

% Plot
figure
semilogy(vfx)
title('Minimization')
xlabel('Iterations')
ylabel('Best f(x,y)')
grid on

txt2 = ['F Best: ', num2str(fbest)];
text(0,1,txt2,'Units','normalized',...
     'HorizontalAlignment','left','VerticalAlignment','bottom');

txt3 = ['F Found: ', num2str(fx)];
text(1,1,txt3,'Units','normalized',...
     'HorizontalAlignment','right','VerticalAlignment','bottom');



% INTERNAL FUNCTIONS

function imprime(PRINT,vx,vy,vz,x,y,fx)

    if PRINT == 1
        meshc(vx,vy,vz)
        hold on
        title('Minimization')
        xlabel('x')
        ylabel('y')
        zlabel('f(x,y)')
        plot3(x,y,fx,'k*')
        colormap jet
        drawnow
        hold off
        pause(0.1)
    end

end

function [T,pcs] = reprod(N,beta,ind,Ab)

    % N	   -> number of clones
    % beta -> multiplying factor
    % ind  -> best individuals
    % Ab   -> antibody population
    
    % T	   -> temporary population
    % pcs  -> final position of each clone
    
    T = [];
    
   for i = 1:N
      cs(i) = round(beta*N);
      pcs(i) = sum(cs);
      T = [T; ones(cs(i),1) * Ab(ind(end-i+1),:)];
   end

end

function pm = pmcont(pm,pma,pmr,it,itpm)

    % pma  -> initial value
    % pmr  -> control rate
    % itpm -> iterations for restoring
    
    if rem(it,itpm) == 0
       pm = pm * pmr;
       if rem(it,10*itpm) == 0
          pm = pma;
       end
    end

end

function z = decode(Ab,varMin,varMax)

    % x	-> real value (precision: 6)
    % v	-> binary string (length: 22)
    
    Ab = fliplr(Ab);
    s = size(Ab);
    aux = 0:1:21;
    aux = ones(s(1),1)*aux;
    x1 = sum((Ab.*2.^aux),2);
    
    % Keeping values between bounds
    z = varMin + x1' .* (varMax - varMin)/(2^22 - 1);

end

function Ab = cadeia(n1,s1)

    % Antibody (Ab) chains
    Ab = 2 .* rand(n1,s1) - 1;
    Ab = hardlim(Ab);

end
