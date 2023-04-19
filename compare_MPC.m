% 定义系统动态方程
A = [0 1; 0 0];
B = [0; 1];
nx = size(A,2);
nu = size(B,2);
Ts = 0.01;

% 定义MPC参数
N = 10; % 控制时域长度
Q = diag([10 1]); % 状态惩罚矩阵
R = 0.1; % 输入惩罚
Qf = Q; % 终端惩罚
x0 = [0.1; 0]; % 初始状态

% 定义CBF参数
alpha = 1; % CBF参数
gamma = 1; % CBF参数

% 构建MPC控制器
mpcobj = mpc(A,B,Q,R,Qf,N);
mpcobj.Model.Ts = Ts;

% 构建CBF控制器
x = mpcobj.Model.State;
u = mpcobj.Model.Input;
cbfobj = controlBarrierFunction(x,u,alpha,gamma);

% 模拟实验
tspan = 0:Ts:10;
x = zeros(nx,numel(tspan));
x(:,1) = x0;
u = zeros(nu,numel(tspan));
for k = 1:numel(tspan)-1
    % 计算MPC控制输入
    u(:,k) = mpcobj(x(:,k));
    
    % 检查CBF
    if cbfobj(x(:,k),u(:,k)) < 0
        disp('Warning: Control Barrier Function violated!');
    end
    
    % 系统仿真
    x(:,k+1) = x(:,k) + A*x(:,k)*Ts + B*u(:,k)*Ts;
end

% 绘图
figure;
subplot(2,1,1);
plot(tspan,x(1,:));
ylabel('Angle');
title('Inverted Pendulum Control with MPC and CBF');
subplot(2,1,2);
plot(tspan,u);
ylabel('Force');
xlabel('Time');
