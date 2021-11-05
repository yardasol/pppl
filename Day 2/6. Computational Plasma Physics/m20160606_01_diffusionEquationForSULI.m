function m20160606_01_diffusionEquationForSULI()

% This script solves the diffusion equation
% d T / d t = d^2 T / d x^2 + F(x)
% where F(x) is some forcing function.

% The number of grid points in x is N_x+2,
% where we set T=0 at the first and last points,
% so there are N_x values of T(x,t) to be determined:
N_x = 10;

t_option = 1;
% 1 =  Forward Euler (explicit)
% 2 = Backward Euler (implicit)
% 3 = Trapezoidal rule = Crank-Nicolson (implicit)

x_option = 1;
% 1 = Finite differences
% 2 = Chebyshev

delta_t = 0.004;
% For the demo, we used delta_t = 0.004 with N_x = 10 or 20 points.

% You can either specify the maximum time desired...
%t_max = 0.4;

% or else you can specify the number of time steps desired.
t_max = delta_t*10;

x_min = 0;
x_max = 1;

    function F = forcing(x)
        F = log(x+1);
    end

% *******************************************
% End of input options.
% *******************************************

    function [x, D] = ChebyshevGridAndDifferentiationMatrix(N, xMin, xMax)
        % From the book "Spectral methods in matlab" by Lloyd Trefethen.
        N1=N-1;
        if N1==0, D=0; x=1; return, end
        x = cos(pi*(0:N1)/N1)';
        c = [2; ones(N1-1,1); 2].*(-1).^(0:N1)';
        X = repmat(x,1,N1+1);
        dX = X-X';
        D = (c*(1./c)')./(dX+(eye(N1+1)));  % off-diagonal entries
        D = D - diag(sum(D'));             % diagonal entries
        D = D * 2/(xMax-xMin);
        x = (x+1) * (xMax-xMin)/2 + xMin;
        D=fliplr(flipud(D));
        x=fliplr(x')';
    end

    function p = chebint(fk, x)
        
        %  The function p = chebint(fk, x) computes the polynomial interpolant
        %  of the data (xk, fk), where xk are the Chebyshev nodes.
        %  Two or more data points are assumed.
        %
        %  Input:
        %  fk:  Vector of y-coordinates of data, at Chebyshev points
        %       x(k) = cos((k-1)*pi/(N-1)), k = 1...N.
        %  x:   Vector of x-values where polynomial interpolant is to be evaluated.
        %
        %  Output:
        %  p:    Vector of interpolated values.
        %
        %  The code implements the barycentric formula; see page 252 in
        %  P. Henrici, Essentials of Numerical Analysis, Wiley, 1982.
        %  (Note that if some fk > 1/eps, with eps the machine epsilon,
        %  the value of eps in the code may have to be reduced.)
        
        %  From the DMSuite package
        %  J.A.C. Weideman, S.C. Reddy 1998
        
        fk = fk(:); x = x(:);                    % Make sure data are column vectors.
        
        N = length(fk);
        M = length(x);
        
        xk = sin(pi*[N-1:-2:1-N]'/(2*(N-1)));    % Compute Chebyshev points.
        
        w = ones(N,1).*(-1).^[0:N-1]';          % w = weights for Chebyshev formula
        w(1) = w(1)/2; w(N) = w(N)/2;
        
        D = x(:,ones(1,N)) - xk(:,ones(1,M))';  % Compute quantities x-x(k)
        D = 1./(D+eps*(D==0));                  % and their reciprocals.
        
        p = D*(w.*fk)./(D*w);                   % Evaluate interpolant as
        % matrix-vector products.
    end

N_t = round(t_max / delta_t);

% Make x grid and d^2 / dx^2 operator:
switch x_option
    case 1
        % Finite differences
        x = linspace(x_min, x_max, N_x+2)';
        % Discard first and last points:
        x(1)=[];
        x(end)=[];
        dx = x(2)-x(1);
        % Make the differentiation matrix:
        d2dx2 = (-2*diag(ones(N_x,1)) + diag(ones(N_x-1,1),1) + diag(ones(N_x-1,1),-1)) / (dx*dx);
    case 2
        [x, ddx] = ChebyshevGridAndDifferentiationMatrix(N_x+2, x_min, x_max);
        d2dx2 = ddx*ddx;
        x = x(2:end-1);
        % Throw away first and last rows because we do not enforce the diffusion equation at these grid points.
        % Throw away first and last columns since T=0 at these grid points.
        d2dx2 = d2dx2(2:end-1, 2:end-1);
    otherwise
        error('Invalid x_option')
end

x_with_boundaries = [x_min; x; x_max];

switch t_option
    case 2
        % Assemble matrix needed for backwards Euler:
        matrix = eye(N_x) - delta_t * d2dx2;
    case 3
        % Assemble matrix needed for trapezoid rule:
        matrix = eye(N_x) - 0.5 * delta_t * d2dx2;
end

state_vector = zeros(N_x,1);
F = forcing(x);
figure(1)
clf
t=0;

N_x_fine = 100;
x_fine = linspace(x_min,x_max,N_x_fine);
x_fine2 = fliplr(linspace(-1,1,N_x_fine));

cpu_time = 0;
for timestep = 1:N_t
    t = t + delta_t;
    
    tic
    switch t_option
        case 1
            % Forward Euler
            state_vector = state_vector + delta_t * (d2dx2 * state_vector + F);
        case 2
            % Backward Euler
            right_hand_side = state_vector + delta_t * F;
            state_vector = matrix \ right_hand_side;
        case 3
            % Trapezoidal rule:
            right_hand_side = state_vector + (0.5 * delta_t) * d2dx2 * state_vector + delta_t * F;
            state_vector = matrix \ right_hand_side;
        otherwise
            error('Invalid x_option')
    end
    cpu_time = cpu_time + toc;
    
    state_vector_with_boundaries = [0; state_vector; 0];
    clf
    if x_option==1
        % Linear interpolation
        plot(x_with_boundaries, state_vector_with_boundaries, '.-', 'LineWidth',2)
    else
        % Show data with the polynomial interpolant
        plot(x_fine, chebint(state_vector_with_boundaries, x_fine2), '-', 'LineWidth',2)
    end
    hold on
    plot(x_with_boundaries, state_vector_with_boundaries, '.', 'LineWidth',2,'MarkerSize',30)
    
    set(gca,'FontSize',20)
    ylim([min([0;state_vector]), max([0.06; state_vector])])
    title(sprintf('time = %.5g',t))
    xlabel('x')
    ylabel('T')
    drawnow
    %pause(0.01)
end

fprintf('CPU time required for time advances: %g sec.\n',cpu_time)
end

