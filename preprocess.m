function preprocessed = preprocess(x,alpha)
    [N, C] = size(x);
    temp = zeros(N,C);
    preprocessed = temp;
    % Number of samples to be truncated
    trun_count = alpha*N;
    for i = 1:C
        % Standardization
        x(:,i) = (x(:,i) - mean(x(:,i)))/std(x(:,i));
        figure('color',[1 1 1]);
        subplot(3,1,1);
        plot(x(:,i),'k');
        fprintf('Before truncation: Min : %2.2f, Max: %2.2f\n',min(x),max(x));
        % Truncating top and bottom trun_count number of values
        temp(:,i) = x(:,i);
        [~,idx] = sort(x(:,i));
        % Capping the trun_count number of negative values
        temp(idx(1:trun_count),i) = x(idx(trun_count+1),i);
        % Capping the trun_count number of positive values
        temp(idx(N-trun_count+1:N),i) = x(idx(N-trun_count),i); 
        fprintf('After truncation: Min : %2.2f, Max: %2.2f\n',min(temp),max(temp));
        subplot(3,1,2);
        plot(temp(:,i),'b');
        preprocessed(:,i) = MinMaxNorm(temp(:,i),0,1);
        subplot(3,1,3);
        plot(preprocessed(:,i),'m');
    end
end

