#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 5341

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["SaxoRealTimeWorker.csproj", "SaxoRealTimeWorker/"]
#COPY ["../SaxoRealTimeWorker/SaxoRealTimeWorker.csproj", "SaxoRealTimeWorker/"]
RUN dotnet restore "SaxoRealTimeWorker.csproj"
#RUN dotnet restore "SaxoRealTimeWorker/SaxoRealTimeWorker.csproj"
COPY . .
WORKDIR "/src/SaxoRealTimeWorker"
RUN dotnet build "SaxoRealTimeWorker.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "SaxoRealTimeWorker.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "SaxoRealTimeWorker.dll"]