% 导入数据表
data0 = xlsread('D:\Desktop\ptq_result\cifar100\AlexNet.xlsx','Sheet','B4:G46');
data1 = xlsread('D:\Desktop\ptq_result\cifar100\AlexNet_BN.xlsx','Sheet','B4:G46');
data2 = xlsread('D:\Desktop\ptq_result\cifar100\VGG_16.xlsx','Sheet','B4:G46');
data3 = xlsread('D:\Desktop\ptq_result\cifar100\VGG_19.xlsx','Sheet','B4:G46');
data4 = xlsread('D:\Desktop\ptq_result\cifar100\Inception_BN.xlsx','Sheet','B4:G46');
data5 = xlsread('D:\Desktop\ptq_result\cifar100\ResNet_18.xlsx','Sheet','B4:G46');
data6 = xlsread('D:\Desktop\ptq_result\cifar100\ResNet_50.xlsx','Sheet','B4:G46');
data7 = xlsread('D:\Desktop\ptq_result\cifar100\ResNet_152.xlsx','Sheet','B4:G46');
data8 = xlsread('D:\Desktop\ptq_result\cifar100\MobileNetV2.xlsx','Sheet','B4:G46');
file_data = vertcat(data0,data1,data2,data3,data4,data5,data6,data7,data8);
js_flops = file_data(:,1); 
js_flops_wt_log = file_data(:,2);
js_flops_wt_cbrt = file_data(:,3);
js_param = file_data(:,4);
ptq_acc  = file_data(:,5);
acc_loss = file_data(:,6);

% 指定拟合类别（横坐标）及多项式次数
% fit_type 1/2/3对应js_flops/js_flops_wt_log/js_flops_wt_cbrt
fit_type = 1;
poly = 2;

switch fit_type
    case 1
        x = js_flops;
        xlabel('js\_flops');
    case 2
        x = js_flops_wt_log;
        xlabel('js\_flops\_wt\_log');
    case 3
        x = js_flops_wt_cbrt;
        xlabel('js\_flops\_wt\_cbrt');
end

y = acc_loss;
ylabel('acc\_loss');

% 定义颜色向量和每个数据点所属的类别
% colors = ['r', 'g', 'b','m','o','c','dr','db','lm'];
colors = [
    1 0 0;
    0 1 0;
    0 0 1;
    0 1 1;
    1 0 1;
    1 1 0;
    1 0.5 0;
    0 1 0.5;
    0.5 0 1;
    ];
class = [ones(43,1); 2*ones(43,1); 3*ones(43,1);4*ones(43,1);5*ones(43,1);
        6*ones(43,1); 7*ones(43,1); 8*ones(43,1); 9*ones(43,1);];

% 指定拟合模型
switch poly
    case 2
        rational_model = fittype( '(x<10)*((p1*x.^2 + p2*x + p3) / (q1*x.^2 + q2*x + q3)) + (x>=10)*((p1*10^2 + p2*10 + p3) / (q1*10^2 + q2*10 + q3))',...
            'independent', 'x', 'coefficients', {'p1', 'p2', 'p3', 'q1', 'q2', 'q3'});
    case 3
        rational_model = fittype('(p1*x.^3 + p2*x.^2 + p3*x + p4) / (q1*x.^3 + q2*x.^2 + q3*x + q4)', 'independent', 'x', 'coefficients', {'p1', 'p2', 'p3','p4', 'q1', 'q2', 'q3','q4'});
    case 4
        rational_model = fittype('(p0*x.^4 + p1*x.^3 + p2*x.^2 + p3*x + p4) / (q0*x.^4 + q1*x.^3 + q2*x.^2 + q3*x + q4)', 'independent', 'x', 'coefficients', {'p0', 'p1', 'p2', 'p3','p4','q0', 'q1', 'q2', 'q3','q4'});
end

%初次拟合
[fitresult, gof] = fit(x, y, rational_model);

% 确保拟合结果是单调上升的
tolerance = 0;
x_range = min(x):0.5:max(x);
y_fit = fitresult(x_range);
SSE=gof.sse;
R_square = gof.rsquare; 
RMSE = gof.rmse; 
% R_square = 0; 
while any(diff(y_fit) < tolerance)
% while R_square < 0.85
    % 如果拟合曲线不是单调上升的，重新拟合
    [fitresult, gof] = fit(x, y, rational_model);
    y_fit = fitresult(x_range);
    % 获取评价指标
    SSE=gof.sse;
    R_square = gof.rsquare; 
    RMSE = gof.rmse; 
end

% 可视化数据点和拟合曲线
hold on;
for i = 1:9
    color = colors(i,:);
    scatter(x(43*i-42:43*i), y(43*i-42:43*i), [], color, 'filled');
end
% scatter(x(44:86), y(44:86), [], colors(2), 'filled');
% scatter(x(86:129), y(86:129), [], colors(3), 'filled');
% scatter(x(130:172), y(130:172), [], colors(4), 'filled');
% scatter(x(173:215), y(173:215), [], colors(5), 'filled');

plot(fitresult,'k',x,y);

legend('AlexNet', 'AlexNet\_BN','VGG\_16','VGG\_19','Inception\_BN',...
    'ResNet\_18','ResNet\_50','ResNet\_152','MobileNetV2',...
    'ALL', 'Rational-Fit', 'Location', 'Northeast');

% 将拟合公式和 R 方显示在图上
text(0.65, 0.2, sprintf('Goodness of fit:\n SSE:%.4f\n R-square:%.4f\n RMSE:%.4f', SSE, R_square, RMSE), 'Units', 'normalized', 'FontSize', 11);

hold off;