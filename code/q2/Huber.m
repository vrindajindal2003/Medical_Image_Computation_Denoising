function [value,grad] = Huber(x_left,x_right,x_up,x_down,x,y,alpha,gamma)
    value = zeros(256,256);
    grad = zeros(256,256);
    
    for i=1:256
        for j=1:256
            if(abs(x_left(i,j)) <= gamma )
                value(i,j) = value(i,j) + 0.5*(x_left(i,j).*x_left(i,j));
                grad(i,j) = grad(i,j) + x_left(i,j);
            end
            
            if(abs(x_left(i,j)) > gamma)
                value(i,j) = value(i,j) + gamma*abs(x_left(i,j)) - 0.5*gamma*gamma;
                grad(i,j) = grad(i,j) + gamma*sign(x_left(i,j));
            end
            
            if(abs(x_right(i,j)) <= gamma )
                value(i,j) = value(i,j) + 0.5*(x_right(i,j).*x_right(i,j));
                grad(i,j) = grad(i,j) + x_right(i,j);
            end
            
            if(abs(x_right(i,j)) > gamma)
                value(i,j) = value(i,j) + gamma*abs(x_right(i,j)) - 0.5*(x_right(i,j)*x_right(i,j));
                grad(i,j) = grad(i,j) + gamma*sign(x_right(i,j));
            end
            
            if(abs(x_up(i,j)) <= gamma )
                value(i,j) = value(i,j) + 0.5*(x_up(i,j).*x_up(i,j));
                grad(i,j) = grad(i,j) + x_up(i,j);
            end
            
            if(abs(x_up(i,j)) > gamma)
                value(i,j) = value(i,j) + gamma*abs(x_up(i,j)) - 0.5*(x_up(i,j)*x_up(i,j));
                grad(i,j) = grad(i,j) + gamma*sign(x_up(i,j));
            end
            
            if(abs(x_down(i,j)) <= gamma )
                value(i,j) = value(i,j) + 0.5*(x_down(i,j).*x_down(i,j));
                grad(i,j) = grad(i,j) + x_down(i,j);
            end
            
            if(abs(x_down(i,j)) > gamma)
                value(i,j) = value(i,j) + gamma*abs(x_down(i,j)) - 0.5*(x_down(i,j)*x_down(i,j));
                grad(i,j) = grad(i,j) + gamma*sign(x_down(i,j));
            end
        end
    end
    
    value = alpha*(value) + (1 - alpha)*((y-x).*(y-x));
    grad = alpha*(grad) + (1 - alpha)*2*(x-y);
    
    
end