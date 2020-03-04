function [quad,grad] = quadratic(x_left,x_right,x_up,x_down,x,y,alpha,gamma)
    quad1 = x_left.*x_left;
    quad2 = x_right.*x_right;
    quad3 = x_up.*x_up;
    quad4 = x_down.*x_down;
    
    quad = alpha*(quad1 + quad2 + quad3 + quad4) + (1 - alpha)*((y-x).*(y-x));
    grad = 2*(1 - alpha)*(x-y) + 2*alpha*(x_left + x_right + x_up + x_down);
end
