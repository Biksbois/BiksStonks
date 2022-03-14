using Autofac;
using Autofac.Extensions.DependencyInjection;
using SaxoRealTimeWorker;
using Serilog;
using System.Data;
using System.Data.SqlClient;
using System.Diagnostics;
using System.Text;

IHost host = Host.CreateDefaultBuilder(args)
    //.UseSerilog()
    .UseServiceProviderFactory(new AutofacServiceProviderFactory())
    .UseSerilog((context, services, configuration) =>
    {
        configuration
            .ReadFrom.Configuration(context.Configuration)
            .ReadFrom.Services(services)
            .Enrich.FromLogContext()
            //.WriteTo.Console()
            .WriteTo.Seq("http://localhost:5341"); // http://localhost:5341
    })
    .ConfigureContainer<ContainerBuilder>((host, builder) =>
    {
        //builder.Register<IDbConnection>(f =>
        //{
        //    var con = new SqlConnection(host.Configuration.GetValue<string>("DataPlatform_ConnectionString"));
        //    con.Open();
        //    return con;
        //}).As<IDbConnection>().AsSelf().InstancePerDependency().ExternallyOwned()
        //    .Named<IDbConnection>("DataPlatformConnection");

    }).ConfigureAppConfiguration((hostingContext, config) =>
    {
        config.Sources.Clear();

        config.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .AddJsonFile("secrets/appsettings.secrets.json", true)
            .AddUserSecrets<Program>(true);
    })
    .ConfigureServices(services =>
    {
        services.AddHostedService<Worker>();
    })
    .Build();


await host.RunAsync();


