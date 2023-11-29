% 导入数据表
data0 = xlsread('D:\Desktop\ptq_result_weighted.xlsx','AlexNet','B4:F46');
data1 = xlsread('D:\Desktop\ptq_result_weighted.xlsx','AlexNet_BN','B4:F46');
data2 = xlsread('D:\Desktop\ptq_result_weighted.xlsx','VGG_16','B4:F46');
data3 = xlsread('D:\Desktop\ptq_result_weighted.xlsx','VGG_19','B4:F46');
data4 = xlsread('D:\Desktop\ptq_result_weighted.xlsx','Inception_BN','B4:F46');
file_data = vertcat(data0,data1,data2,data3,data4)
js_flops = file_data(:,1); 
js_flops_weighted = file_data(:,2);
js_param = file_data(:,3);
%js_param_weighted = file_data(:,4);
ptq_acc  = file_data(:,4);
acc_loss = file_data(:,5);

% 指定横纵坐标及多项式次数
x = js_flops_weighted;
y = acc_loss;
poly = 4;

% 定义颜色向量和每个数据点所属的类别
colors = ['r', 'g', 'b','m','o'];
class = [ones(43,1); 2*ones(43,1); 3*ones(43,1);4*ones(43,1);5*ones(43,1);];

% 指定拟合模型
if poly == 2
    rational_model = fittype('(p1*x.^2 + p2*x + p3) / (q1*x.^2 + q2*x + q3)', 'independent', 'x', 'coefficients', {'p1', 'p2', 'p3', 'q1', 'q2', 'q3'});
elseif poly == 3
    rational_model = fittype('(p1*x.^3 + p2*x.^2 + p3*x + p4) / (q1*x.^3 + q2*x.^2 + q3*x + q4)', 'independent', 'x', 'coefficients', {'p1', 'p2', 'p3','p4', 'q1', 'q2', 'q3','q4'});
elseif poly == 4
    rational_model = fittype('(p0*x.^4 + p1*x.^3 + p2*x.^2 + p3*x + p4) / (q0*x.^4 + q1*x.^3 + q2*x.^2 + q3*x + q4)', 'independent', 'x', 'coefficients', {'p0', 'p1', 'p2', 'p3','p4','q0', 'q1', 'q2', 'q3','q4'});
end

%初次拟合
[fitresult, gof] = fit(x, y, rational_model);

% 确保拟合结果是单调上升的
tolerance = 0;
x_range = min(x):0.1:max(x);
y_fit = fitresult(x_range);
while any(diff(y_fit) < tolerance)
    % 如果拟合曲线不是单调上升的，重新拟合
    [fitresult, gof] = fit(x, y, rational_model);
    y_fit = fitresult(x_range);
end


% 可视化数据点和拟合曲线
scatter(x(1:43), y(1:43), [], colors(1), 'filled');
hold on;
scatter(x(44:86), y(44:86), [], colors(2), 'filled');
scatter(x(86:129), y(86:129), [], colors(3), 'filled');
scatter(x(130:172), y(130:172), [], colors(4), 'filled');
scatter(x(173:215), y(173:215), [], colors(5), 'filled');

plot(fitresult,'k',x,y);
xlabel('js\_flops\_weighted');
ylabel('acc\_loss')
legend('AlexNet', 'AlexNet\_BN','VGG\_16','VGG\_19','Inception\_BN','ALL', 'Rational-Fit', 'Location', 'Northeast');
% 获取评价指标
SSE=gof.sse;
R_square = gof.rsquare; 
RMSE = gof.rmse; 

% 将拟合公式和 R 方显示在图上
text(0.65, 0.2, sprintf('Goodness of fit:\n SSE:%.4f\n R-square:%.4f\n RMSE:%.4f', SSE, R_square, RMSE), 'Units', 'normalized', 'FontSize', 11);

hold off;