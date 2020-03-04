noisey = imread('../../data/histology_noisy.png');
m=double(noisey)/255;

noiseless = imread('../../data/histology_noiseless.png');
%imshow(noisey);
noiseless = double(noiseless);
noiseless = noiseless/255;

y = double(noisey);
y = y/255;
x_initial = y;
x_initial_r=y(:,:,1);
x_initial_g=y(:,:,2);
x_initial_b=y(:,:,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%QUADRATIC

alpha = 0.7375;
gamma = -1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%opt alpha
[final_xi_quad_1_r,obj_quad_1_r] = gradient_descent( x_initial_r , alpha , @quadratic,gamma);
[final_xi_quad_1_g,obj_quad_1_g] = gradient_descent( x_initial_g , alpha , @quadratic,gamma);
[final_xi_quad_1_b,obj_quad_1_b] = gradient_descent( x_initial_b , alpha , @quadratic,gamma);

final_xi_quad_1=zeros(256,256,3);
final_xi_quad_1(:,:,1)=final_xi_quad_1_r;
final_xi_quad_1(:,:,2)=final_xi_quad_1_g;
final_xi_quad_1(:,:,3)=final_xi_quad_1_b;

    noise=final_xi_quad_1;
    a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    
    rrmse_quad_1=top/bot;
    
  %%%%%%%%%%%%%%%%%%%%%%%%%0.8*alpha%%%%%%%%%%%%%%%%%
  [final_xi_quad_2_r,obj_quad_2_r] = gradient_descent( x_initial_r , 0.8*alpha , @quadratic,gamma);
[final_xi_quad_2_g,obj_quad_2_g] = gradient_descent( x_initial_g , 0.8*alpha , @quadratic,gamma);
[final_xi_quad_2_b,obj_quad_2_b] = gradient_descent( x_initial_b , 0.8*alpha , @quadratic,gamma);

final_xi_quad_2=zeros(256,256,3);
final_xi_quad_2(:,:,1)=final_xi_quad_2_r;
final_xi_quad_2(:,:,2)=final_xi_quad_2_g;
final_xi_quad_2(:,:,3)=final_xi_quad_2_b;

    noise=final_xi_quad_2;
    a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    
    rrmse_quad_2=top/bot;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%51.2*alpha%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    
      [final_xi_quad_3_r,obj_quad_3_r] = gradient_descent( x_initial_r , 1.2*alpha , @quadratic,gamma);
[final_xi_quad_3_g,obj_quad_3_g] = gradient_descent( x_initial_g , 1.2*alpha , @quadratic,gamma);
[final_xi_quad_3_b,obj_quad_3_b] = gradient_descent( x_initial_b , 1.2*alpha , @quadratic,gamma);

final_xi_quad_3=zeros(256,256,3);
final_xi_quad_3(:,:,1)=final_xi_quad_3_r;
final_xi_quad_3(:,:,2)=final_xi_quad_3_g;
final_xi_quad_3(:,:,3)=final_xi_quad_3_b;

    noise=final_xi_quad_3;
    a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    
    rrmse_quad_3=top/bot;
    
    %%%%%%%%%%%%%%%%%%%%print for quad%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');

fprintf('Denoising Using Quadratic Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('RRMSE(alpha) = %f\n', rrmse_quad_1);
fprintf('RRMSE(alpha*0.8) = %f\n' , rrmse_quad_2);
fprintf('RRMSE(alpha*1.2) = %f\n' , rrmse_quad_3);
fprintf('\n');

figure(1);
plot(obj_quad_1_r);
title('Quadratic Denoising for RED channel');
xlabel('No. of iterations');
ylabel('Cost Function')

figure(2);
plot(obj_quad_1_g);
title('Quadratic Denoising for GREEN channel');
xlabel('No. of iterations');
ylabel('Cost Function')

figure(3);
plot(obj_quad_3_b);
title('Quadratic Denoising for BLUE channel');
xlabel('No. of iterations');
ylabel('Cost Function')
%end quadratic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Huber%%%%%%%%%%%%%%%%%%%%%5
alpha=0.8400;
gamma=0.0600*1.2;
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%OPTIMAL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[final_xi_huber_1_r,obj_huber_1_r] = gradient_descent( x_initial_r , alpha , @Huber,gamma);
[final_xi_huber_1_g,obj_huber_1_g] = gradient_descent( x_initial_g , alpha , @Huber,gamma);
[final_xi_huber_1_b,obj_huber_1_b] = gradient_descent( x_initial_b , alpha , @Huber,gamma);

final_xi_huber_1=zeros(256,256,3);
final_xi_huber_1(:,:,1)=final_xi_huber_1_r;
final_xi_huber_1(:,:,2)=final_xi_huber_1_g;
final_xi_huber_1(:,:,3)=final_xi_huber_1_b;

noise=final_xi_huber_1;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_huber_1=top/bot;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%0.8alpha%%%%%%%%%%%%%%%%%%%%%%%
    [final_xi_huber_2_r,obj_huber_2_r] = gradient_descent( x_initial_r , 0.8*alpha , @Huber,gamma);
[final_xi_huber_2_g,obj_huber_2_g] = gradient_descent( x_initial_g , 0.8*alpha , @Huber,gamma);
[final_xi_huber_2_b,obj_huber_2_b] = gradient_descent( x_initial_b , 0.8*alpha , @Huber,gamma);

final_xi_huber_2=zeros(256,256,3);
final_xi_huber_2(:,:,1)=final_xi_huber_2_r;
final_xi_huber_2(:,:,2)=final_xi_huber_2_g;
final_xi_huber_2(:,:,3)=final_xi_huber_2_b;

noise=final_xi_huber_2;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_huber_2=top/bot;
    %%%%%%%%%%%%%%%%%%%%%%%%%alpha*1.2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
       [final_xi_huber_3_r,obj_huber_3_r] = gradient_descent( x_initial_r , 1.2*alpha , @Huber,gamma);
[final_xi_huber_3_g,obj_huber_3_g] = gradient_descent( x_initial_g , 1.2*alpha , @Huber,gamma);
[final_xi_huber_3_b,obj_huber_3_b] = gradient_descent( x_initial_b , 1.2*alpha , @Huber,gamma);

final_xi_huber_3=zeros(256,256,3);
final_xi_huber_3(:,:,1)=final_xi_huber_3_r;
final_xi_huber_3(:,:,2)=final_xi_huber_3_g;
final_xi_huber_3(:,:,3)=final_xi_huber_3_b;

noise=final_xi_huber_3;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_huber_3=top/bot;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%0.8gamma%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       [final_xi_huber_4_r,obj_huber_4_r] = gradient_descent( x_initial_r , alpha , @Huber,0.8*gamma);
[final_xi_huber_4_g,obj_huber_4_g] = gradient_descent( x_initial_g , alpha , @Huber,0.8*gamma);
[final_xi_huber_4_b,obj_huber_4_b] = gradient_descent( x_initial_b , alpha , @Huber,0.8*gamma);

final_xi_huber_4=zeros(256,256,3);
final_xi_huber_4(:,:,1)=final_xi_huber_4_r;
final_xi_huber_4(:,:,2)=final_xi_huber_4_g;
final_xi_huber_4(:,:,3)=final_xi_huber_4_b;

noise=final_xi_huber_4;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_huber_4=top/bot;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%1.2gamma%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [final_xi_huber_5_r,obj_huber_5_r] = gradient_descent( x_initial_r , alpha , @Huber,1.2*gamma);
[final_xi_huber_5_g,obj_huber_5_g] = gradient_descent( x_initial_g , alpha , @Huber,1.2*gamma);
[final_xi_huber_5_b,obj_huber_5_b] = gradient_descent( x_initial_b , alpha , @Huber,1.2*gamma);

final_xi_huber_5=zeros(256,256,3);
final_xi_huber_5(:,:,1)=final_xi_huber_5_r;
final_xi_huber_5(:,:,2)=final_xi_huber_5_g;
final_xi_huber_5(:,:,3)=final_xi_huber_5_b;

noise=final_xi_huber_5;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_huber_5=top/bot;
    
    
    fprintf('Denoising High noise Using Huber Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('Optimal Gamma = %f\n' , gamma);
fprintf('RRMSE(alpha,gamma) = %f\n', rrmse_huber_1);
fprintf('RRMSE(alpha*0.8,gamma) = %f\n' , rrmse_huber_2);
fprintf('RRMSE(alpha*1.2,gamma) = %f\n' , rrmse_huber_3);
fprintf('RRMSE(alpha,gamma*0.8) = %f\n' , rrmse_huber_4);
fprintf('RRMSE(alpha,gamma*1.2) = %f\n' , rrmse_huber_5);

fprintf('\n');

figure(4);
plot(obj_huber_1_r);
title('Huber Denoising in RED noise');
xlabel('No. of iterations');
ylabel('Cost Function')

figure(5);
plot(obj_huber_1_g);
title('Huber Denoising in GREEN noise');
xlabel('No. of iterations');
ylabel('Cost Function')

figure(6);
plot(obj_huber_1_b);
title('Huber Denoising in BLUE noise');
xlabel('No. of iterations');
ylabel('Cost Function')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%end huber%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%disc



        alpha=0.8300;
        gamma=19.0;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%opt
 
[final_xi_disc_1_r,obj_disc_1_r] = gradient_descent(x_initial_r , alpha, @Disc,gamma);
[final_xi_disc_1_g,obj_disc_1_g] = gradient_descent(x_initial_g , alpha, @Disc,gamma);
[final_xi_disc_1_b,obj_disc_1_b] = gradient_descent(x_initial_b , alpha, @Disc,gamma);

final_xi_disc_1=zeros(256,256,3);
final_xi_disc_1(:,:,1)=final_xi_disc_1_r;
final_xi_disc_1(:,:,2)=final_xi_disc_1_g;
final_xi_disc_1(:,:,3)=final_xi_disc_1_b;

noise=final_xi_disc_1;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_disc_1=top/bot;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%0.8alpha%%%%%%%%%%%%%%%%%%%%%%%555
    [final_xi_disc_2_r,obj_disc_2_r] = gradient_descent(x_initial_r , 0.8*alpha, @Disc,gamma);
[final_xi_disc_2_g,obj_disc_2_g] = gradient_descent(x_initial_g , 0.8*alpha, @Disc,gamma);
[final_xi_disc_2_b,obj_disc_2_b] = gradient_descent(x_initial_b , 0.8*alpha, @Disc,gamma);

final_xi_disc_2=zeros(256,256,3);
final_xi_disc_2(:,:,1)=final_xi_disc_2_r;
final_xi_disc_2(:,:,2)=final_xi_disc_2_g;
final_xi_disc_2(:,:,3)=final_xi_disc_2_b;

noise=final_xi_disc_2;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_disc_2=top/bot;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%1.2LPHA%%%%%%%%%%%%%%%%55
    [final_xi_disc_3_r,obj_disc_3_r] = gradient_descent(x_initial_r , 1.2*alpha, @Disc,gamma);
[final_xi_disc_3_g,obj_disc_3_g] = gradient_descent(x_initial_g , 1.2*alpha, @Disc,gamma);
[final_xi_disc_3_b,obj_disc_3_b] = gradient_descent(x_initial_b , 1.2*alpha, @Disc,gamma);

final_xi_disc_3=zeros(256,256,3);
final_xi_disc_3(:,:,1)=final_xi_disc_3_r;
final_xi_disc_3(:,:,2)=final_xi_disc_3_g;
final_xi_disc_3(:,:,3)=final_xi_disc_3_b;

noise=final_xi_disc_3;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_disc_3=top/bot;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%0.8gamma%%%%%%%%%%%%%%%%%%%
    [final_xi_disc_4_r,obj_disc_4_r] = gradient_descent(x_initial_r , alpha, @Disc,0.8*gamma);
[final_xi_disc_4_g,obj_disc_4_g] = gradient_descent(x_initial_g , alpha, @Disc,0.8*gamma);
[final_xi_disc_4_b,obj_disc_4_b] = gradient_descent(x_initial_b , alpha, @Disc,0.8*gamma);

final_xi_disc_4=zeros(256,256,3);
final_xi_disc_4(:,:,1)=final_xi_disc_4_r;
final_xi_disc_4(:,:,2)=final_xi_disc_4_g;
final_xi_disc_4(:,:,3)=final_xi_disc_4_b;

noise=final_xi_disc_4;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_disc_4=top/bot;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%1.2gamma%%%%%%%%%%%%%%%%%%%5
    [final_xi_disc_5_r,obj_disc_5_r] = gradient_descent(x_initial_r , alpha, @Disc,1.2*gamma);
[final_xi_disc_5_g,obj_disc_5_g] = gradient_descent(x_initial_g , alpha, @Disc,1.2*gamma);
[final_xi_disc_5_b,obj_disc_5_b] = gradient_descent(x_initial_b , alpha, @Disc,1.2*gamma);

final_xi_disc_5=zeros(256,256,3);
final_xi_disc_5(:,:,1)=final_xi_disc_5_r;
final_xi_disc_5(:,:,2)=final_xi_disc_5_g;
final_xi_disc_5(:,:,3)=final_xi_disc_5_b;

noise=final_xi_disc_5;
a=abs(noiseless)-abs(noise);
    a=a.*a;
    a=sum(sum(sum(double(a))));
    top=sqrt(a);

    b=noiseless.*noiseless;
    b=sum(sum(sum(double(b))));
    bot=sqrt(b);
    rrmse_disc_5=top/bot;
    
    
    
    fprintf('Denoising High noise Using Discontinuity-adaptive-log Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('Optimal Gamma = %f\n' , gamma);
fprintf('RRMSE(alpha,gamma) = %f\n', rrmse_disc_1);
fprintf('RRMSE(alpha*0.8,gamma) = %f\n' , rrmse_disc_2);
fprintf('RRMSE(alpha*1.2,gamma) = %f\n' , rrmse_disc_3);
fprintf('RRMSE(alpha,gamma*0.8) = %f\n' , rrmse_disc_4);
fprintf('RRMSE(alpha,gamma*1.2) = %f\n' , rrmse_disc_5);

fprintf('\n');

figure(7);
plot(obj_disc_1_r);
title('LogPrior Denoising in RED noise ');
xlabel('No. of iterations');
ylabel('Cost Function')

figure(8);
plot(obj_disc_1_g);
title('LogPrior Denoising in GREEN noise ');
xlabel('No. of iterations');
ylabel('Cost Function')

figure(9);
plot(obj_disc_1_b);
title('LogPrior Denoising in BLUE noise ');
xlabel('No. of iterations');
ylabel('Cost Function')


 

% 
figure(10);


myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (noiseless)); % phantom is a popular test image
colormap (myColorScale);
colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar;
title('Noiseless');
% 
% subplot(2,3,2);
figure(11);

myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (y)); % phantom is a popular test image
colormap (myColorScale);
colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar;
title('Noisey');
% 
figure(12)
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_quad_1)); % phantom is a popular test image
colormap (myColorScale);
colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar;
title('Quadratic Potential');
% 
figure(13)
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_huber_1)); % phantom is a popular test image
colormap (myColorScale);
colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar;
title('Huber Potential');
% 
figure(14)
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_disc_1)); % phantom is a popular test image
colormap (myColorScale);
colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar;
title('Discontinuity Removal');
% 
% 
% 

