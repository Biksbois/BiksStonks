using Autofac;
using Autofac.Extensions.DependencyInjection;
using SaxoRealTimeData;
using Serilog;
using Serilog.Sinks.Grafana.Loki;

var builder = Host.CreateDefaultBuilder(args)
    .UseSerilog()
    .UseServiceProviderFactory(new AutofacServiceProviderFactory())
    .ConfigureContainer<ContainerBuilder>((host, builder) =>
    {
        Serilog.ILogger logger = new LoggerConfiguration().WriteTo.GrafanaLoki("http://localhost:3100").CreateLogger();

        builder.RegisterInstance(logger);
    })
    .ConfigureServices(services =>
    {
        services.AddHostedService<SaxoWorker>();
    });

var app = builder.Build();

app.Run();
