using Plots
using Plots.PlotMeasures
Plots.plotlyjs()

plotPath!(f,θ,l,col = :green) = plot3d!(first.(θ),last.(θ),f.(θ), label=l, c = col)
function plotPathXY(R)
    x = first.(R); y = last.(R)
    plot(x,y)
    scatter!(x,y)
end

function plotPathXY!(R, col = :blue, lbl = "")
    x = first.(R); y = last.(R)
    plot!(x,y, c = col, label = lbl)
    scatter!(x,y, c = col, label = "", markersize = 2, 
                                       markershape = :diamond)
end

function plotSurface(f;xrange = [-1,1],yrange = [-1,1],l=500)
    xmin,xmax = xrange
    ymin,ymax = yrange
    x=range(xmin,stop=xmax,length=l)
    y=range(ymin,stop=ymax,length=l)

    plot(x,y,f,st = :surface,
                c = cgrad(:thermal, 12, categorical = true, scale = :exp),
                xlabel = "t1",
                ylabel = "t2", 
                zlabel = "L(θ)")
end
