function beta = controlBarrierFunction(x,u,alpha,gamma)
% 控制边界函数
%
% x: 状态变量
% u: 控制输入
% alpha: CBF参数
% gamma: CBF参数
%
% beta: CBF值

% 计算CBF
h = x(1);
dh = x(2);
beta = dh + alpha*h + gamma*abs(u);

% 检查CBF是否被满足
if beta >= 0
    disp('Warning: Control Barrier Function violated!');
end
