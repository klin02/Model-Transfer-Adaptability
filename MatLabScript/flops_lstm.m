% 导入数据表
file_data = xlsread('D:\Desktop\ptq_result.xlsx','PTB_LSTM','B4:G46');
js_flops = file_data(:,1); 
js_param = file_data(:,4);
ptq_acc  = file_data(:,5);
ppl_ratio = file_data(:,6);

x = js_flops;
y = ppl_ratio;

% 定义颜色向量和每个数据点所属的类别
colors = ['r', 'g', 'm'];
class = [ones(16,1); 2*ones(6,1); 3*ones(21,1)];

% 指定拟合模型
rational_model = fittype('(p1*js_flops.^2 + p2*js_flops + p3) / (q1*js_flops.^2 + q2*js_flops + q3)', 'independent', 'js_flops', 'coefficients', {'p1', 'p2', 'p3', 'q1', 'q2', 'q3'});

%初次拟合
[fitresult, gof] = fit(x, y, rational_model);

% 确保拟合结果是单调上升的
tolerance = 0;
x_range = min(x):5:max(x);
y_fit = fitresult(x_range);
while any(diff(y_fit) < tolerance)
    % 如果拟合曲线不是单调上升的，重新拟合
    [fitresult, gof] = fit(x, y, rational_model);
    y_fit = fitresult(x_range);
end

% 可视化数据点和拟合曲线
scatter(x(1:15), y(1:15), [], colors(1), 'filled');
hold on;
scatter(x(16:22), y(16:22), [], colors(2), 'filled');
scatter(x(23:43), y(23:43), [], colors(3), 'filled');
plot(fitresult,'k',x,y);
xlabel('js\_flops');
ylabel('loss\_ratio');
legend('INT', 'POT', 'FLOAT','ALL', 'Rational-Fit', 'Location', 'Northeast');
% 获取评价指标
SSE=gof.sse;
R_square = gof.rsquare; 
RMSE = gof.rmse; 

% 将拟合公式和 R 方显示在图上
text(0.65, 0.2, sprintf('Goodness of fit:\n SSE:%.4f\n R-square:%.4f\n RMSE:%.4f', SSE, R_square, RMSE), 'Units', 'normalized', 'FontSize', 11);

hold off;