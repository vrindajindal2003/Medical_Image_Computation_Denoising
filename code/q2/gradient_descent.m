function [final_xi,obj_fun] = gradient_descent(x_initial,alpha,f,gamma)
%     f
    y = x_initial;
    x = x_initial;
    step_size = 1e-4;
    new_ans = zeros(256,256);
    new = 0;
    x_left = x - circshift(x,1,2);
    x_right = x - circshift(x,-1,2);
    x_up = x - circshift(x,1,1); 
    x_down = x - circshift(x,-1,1);
    no_of_iterations = 0;
    
%     f
    [prev_ans,grad] = f(x_left , x_right , x_up , x_down , x , y , alpha,gamma);
    
    prev = norm(double(prev_ans),'fro');    
    %prev_ans = norm(double(mat));
    obj_fun = zeros(0);
    while step_size > 1e-8;
        %prev_ans = new_ans;
       
        x_left = x - circshift(x,1,2);
        x_right = x - circshift(x,-1,2);
        x_up = x - circshift(x,1,1); 
        x_down = x - circshift(x,-1,1);
        [ans,grad] = f(x_left , x_right , x_up , x_down , x , y , alpha,gamma);     
        %grad = 2*alpha*(x - y) + 2*(x - x_left) + 2*(x - x_right) + 2*(x - x_up) + 2*(x - x_down);
        
        x_dash = x - step_size*grad;
        
        x_left = x_dash - circshift(x_dash,1,2);
        x_right = x_dash - circshift(x_dash,-1,2);
        x_up = x_dash - circshift(x_dash,1,1); 
        x_down = x_dash - circshift(x_dash,-1,1);
        
        
        [new_ans,new_grad] =  f(x_left , x_right , x_up , x_down , x_dash , y , alpha,gamma);

        new = norm(double(new_ans),'fro');
        %prev = norm(double(prev_ans));
        if new >= prev
           % x = x + step_size*grad;
            step_size = 0.5*step_size;
            obj_fun = [obj_fun ; prev];
        end
        
        if new < prev
            step_size = 1.1*step_size;
            obj_fun = [obj_fun ; new];
            x = x_dash;
             prev = new;
        end
        
        no_of_iterations = no_of_iterations + 1;
        
    end
%     f
    final_xi = x;
%     disp(no_of_iterations);
    
    
   % plot(obj_fun)
end