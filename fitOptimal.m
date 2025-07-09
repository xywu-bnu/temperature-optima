function [a, Topt, GPPopt, Topt_exist, R2, RMSE, p1, p2, p3] = fitOptimal(T, GPP, interval)
    % fitOptimal fits a quadratic model between temperature (T) and GPP,
    % calculates the optimal temperature (Topt) and GPP at Topt (GPPopt),
    % and evaluates model performance.
    %
    % Inputs:
    %   T        - Vector of temperature values
    %   GPP      - Vector of corresponding GPP values
    %   interval - Temperature interval 
    %
    % Outputs:
    %   a          - Quadratic coefficient (T^2 term)
    %   Topt       - Optimal temperature (vertex of parabola)
    %   GPPopt     - Maximum GPP value at Topt
    %   Topt_exist - Topt if it lies within observed range, else NaN
    %   R2         - Coefficient of determination
    %   RMSE       - Root Mean Square Error
    %   p1, p2, p3 - p-values of the intercept, linear, and quadratic terms

    % Save original T and GPP for later use
    Torgn = T;
    GPPorgn = GPP;
    
    % Remove invalid data where T < 0
    rowsToRemove = T < 0;
    GPP = GPP(~rowsToRemove);
    T = T(~rowsToRemove);
    
    % Remove rows with NaN or empty GPP values
    rowsToRemove2 = isnan(GPP) | isempty(GPP);
    GPP = GPP(~rowsToRemove2);
    T = T(~rowsToRemove2);
    
    if length(T) > 3
        % Apply moving average smoothing over 3 points
        GPP = movmean(GPP, 3);
        T = movmean(T, 3);
    
        % Fit a quadratic polynomial (2nd degree)
        p = polyfit(T, GPP, 2);
        a = p(1);
        b = p(2);
        c = p(3);
    
        % Calculate vertex of the parabola (Topt and GPPopt)
        Topt = -b / (2 * a);
        GPPopt = c - b^2 / (4 * a);
    
        % Check if Topt is within the observed T range (+/- 0.5*interval)
        tmax = max(Torgn) + 0.5 * interval;
        tmin = min(Torgn) - 0.5 * interval;
        isWithinRange = Topt > tmin && Topt < tmax;
        
        if a < 0 && isWithinRange
            % Topt exists and parabola opens downward
            Topt_exist = Topt;
    
            % Evaluate fit quality
            GPP_fit = polyval(p, T); % Predicted GPP values
            SS_total = sum((GPP - mean(GPP)).^2);
            SS_residual = sum((GPP - GPP_fit).^2);
            R2 = 1 - SS_residual / SS_total; % Coefficient of determination
            N = length(GPP);
            RMSE = sqrt(sum((GPP - GPP_fit).^2) / N); % Root Mean Square Error
    
            % Perform linear regression for p-values
            X = [ones(length(T), 1), T, T.^2]; % Design matrix
            stats = regstats(GPP, X, 'linear', {'tstat'});
            pValues = stats.tstat.pval; % p-values of coefficients
        else
            % Topt does not exist within the range or parabola opens upward
            Topt_exist = nan;
            R2 = nan;
            RMSE = nan;
            pValues = nan(3, 1);
        end
    else
        % Not enough data points to fit the model
        a = nan;
        GPPopt = nan;
        Topt = nan;
        Topt_exist = nan;
        R2 = nan;
        RMSE = nan;
        pValues = nan(3, 1);
    end

    % Assign p-values to outputs
    p1 = pValues(1);
    p2 = pValues(2);
    p3 = pValues(3);
end
