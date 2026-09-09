function z = myfunc(xx)
 N = size(xx,1);
    z = zeros(N,1);

    lambda = 1;   % 权重，控制平均转矩与转矩脉动的权衡
    inp=3.5;

    for i = 1:N
        param = xx(i,1);   % 第 i 个方案的设计参数

        % 1. 用这个参数跑一遍 FEMM，得到转矩指标
        femmfunc(param);%这里先画图
        [T_avg, T_ripple] = scantorque(inp);%这里计算转矩

        % 2. 定义一个"越小越好"的目标 z(i)
        %    举个例子：我们想让 (T_avg - lambda*T_ripple) 尽量大
        F = T_avg - lambda * T_ripple;

        % 免疫算法是最小化问题，所以 z = -F
        z(i) = -F;
    end
end