function nw = MinMaxNorm(w, newmin, newmax)
    minw = min(w);
    maxw = max(w);
    range1 = maxw - minw; % Input Range
    nw = (w-minw)./range1; % Normalized to [0,1]
    range2 = newmax - newmin; % New range [newmin,newmax]
    nw = (nw*range2)+ newmin; % Normalized to range2
end

