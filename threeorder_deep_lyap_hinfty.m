% 定义系统动态方程
A = [0 1 0 0; 0 0 -1 0; 0 0 0 1; 0 0 4.9 0];
B = [0; 1; 0; -1];
nx = size(A,2);
nu = size(B,2);
Ts = 0.01;

% 定义深层展开控制器
layers = [    imageInputLayer([nx, 1, 1],'Name','state')
    fullyConnectedLayer(32,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(32,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(32,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(nu,'Name','output')];
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);

% 定义李雅普诺夫稳定性控制器和$h_\infty$控制器
Q = eye(nx);
R = eye(nu);
S = eye(nx);
G = dlqr(A,B,Q,R,S);
H = ss(A-B*G,zeros(nx,1),eye(nx),0,Ts);
W = tf([1 0.3],[1 -0.8],Ts);

% 定义损失函数和优化器
objective = @(Y, T) mse(Y - T);
opt = adam(1e-3);

% 训练深层展开控制器
N = 10000; % 训练样本数量
X = rand(nx,N);
Y = zeros(nu,N);
for k = 1:N
    Y(:,k) = dlnet.predict(X(:,k));
end
T = zeros(nu,N);
for k = 1:N
    T(:,k) = -A*X(:,k) + B*Y(:,k);
end
dlnet = trainNetwork(X,dlnet,T,objective,opt,'MaxEpochs',50);

% 模拟实验
x = [0.1; 0; 0; 0];
u = zeros(nu,numel(tspan));
for k = 1:numel(tspan)-1
    % 计算深层展开控制输入
    dlx = dlarray(x,'SSCB');
    dlu = predict(dlnet,dlx);
    u(:,k) = extractdata(dlu);
    
    % 添加噪声干扰
    d = 0.01*randn(1);
    y = C*x + d;
    
    % 李雅普诺夫稳定性控制
    xhat = x - H.C*H.State;
    u(:,k) = -G*xhat + H*Dy*W*y;
    
    % h无穷控制
    d = d + 0.1*randn(1);
    u(:,k) = lsim(W,d,tspan(1:k),u(:,1:k-1)) + G*xhat;
    % 计算下一时刻状态
    xdot = A*x + B*u(:,k);
    x = x + Ts*xdot;
end

% 绘制结果
figure;
plot(tspan,u);
xlabel('Time (s)');
ylabel('Control input');
title('Deep Unfolding Control with Robustness Enhancements');
