function [value,grad] = Disc(x_left,x_right,x_up,x_down,x,y,alpha,gamma)
    
    grad = zeros(217,181);
    value1 = gamma*abs(x_left) - gamma*gamma*log(1 + abs(x_left)/gamma);
    value2 = gamma*abs(x_right) - gamma*gamma*log(1 + abs(x_right)/gamma);
    value3 = gamma*abs(x_up) - gamma*gamma*log(1 + abs(x_up)/gamma);
    value4 = gamma*abs(x_down) - gamma*gamma*log(1 + abs(x_down)/gamma);
    value = alpha*(value1 + value2 + value3 + value4) + (1 - alpha)*((y-x).*(y-x));
    
    grad = grad + gamma*x_left./(gamma + abs(x_left));
    grad = grad + gamma*x_right./(gamma + abs(x_right));
    grad = grad + gamma*x_up./(gamma + abs(x_up));
    grad = grad + gamma*x_down./(gamma + abs(x_down));
    
    grad = alpha*grad + (1 - alpha)*(2)*(x-y);
end