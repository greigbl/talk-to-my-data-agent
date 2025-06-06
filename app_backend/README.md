# Talk to My Data: FastAPI App app_backend

## Tech Stack

- FastAPI
- React 18 with TypeScript served as static files

## Development

Start the backend from the project root:

```bash
make run-local-dev-backend
```

To start the development server, cd into the `app_frontend` directory first:

```bash
npm i
npm run dev
```

Make sure to open the frontend url (likely on vite default port 5173) _not_ the backend url.

### Known issue with Apple Silicon (arm64)

If you're on an Apple Silicon (arm64) machine, you might encounter issues with optional dependencies or platform-specific packages. If you encounter errors after a teammate updates `package-lock.json` on a different architecture, delete `node_modules` and `package-lock.json`, then run `npm install` again.

```bash
rm -rf node_modules package-lock.json
npm i
npm run dev
```

## Building

To build the application for production:

```bash
npm run build
```

The build output will be placed in the `../app_backend/static/` directory, which is then used by the Python backend to serve the application. When using the React frontend through the `FRONTEND_TYPE="react"` environment variable, the application will look for the built files in this location.

You can run it locally using the static files by starting it from project root with:

```bash
make run-local-static-backend
```

## Codespaces

### Embedded Codespace (App Source > 'Open in Codespace')

In this type of codespace only the frontend can only be served as static files. Python files can be modified in this edit mode.
To run the app:

```bash
python -m pip install -r requirements.txt
chmod +x start-app.sh
./start-app.sh
```

### Codespace (Workbench > Notebooks & codespaces)

#### Static Frontend

Before starting the codespace, make sure that the 8080 port is opened in `Session environment` settings.
To run the app with prebuilt frontend in a codespace, first open a terminal and prepare the frontend files:

```bash
cd ~/storage/app_frontend
npm install
npm run build:codespace
```

#### _Note:_ When you're ready to publish a new version of the application using Pulumi, make sure to build the static/assets by running this command:

```bash
cd ~/storage/app_frontend
npm run build
```

Next install the necessary python dependencies:

```bash
cd ~/storage/app_backend
python -m pip install -r requirements.txt
```

Now start the app with:

```bash
PORT=8080 DEV_MODE=True python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level info --proxy-headers --reload
```

Navigate to `Session environment` and find the link to the app in the exposed ports section.

#### Dev mode Frontend

For this approach you will need to use a compatible execution environment with NodeJS.
Go to `Session environment` and select `Python 3.11`, you will
also need to open up ports 8080 and 5173

To run the app with a development frontend server in a codespace, first open a terminal and install the necessary python dependencies:

```bash
cd ~/storage/app_backend
python -m pip install -r requirements.txt
```

Then start the app with:

```bash
SERVE_STATIC_FRONTEND=False DEV_MODE=True python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level info --proxy-headers --reload
```

Prepare and run the frontend development server:

```bash
cd ~/storage/app_frontend
npm install
npm run dev
```

Navigate to `Session environment` and find the frontend app link on port 5173 in the exposed ports section.

## Testing

To run tests:

```
uv run pytest --cov --cov-report term --cov-report html
```

To test through a DataRobot like proxy:

Install traefik by downloading from: https://github.com/traefik/traefik/releases

Configure it like with `traefik.toml` like:

```toml
[entryPoints]
  [entryPoints.http]
    address = ":9999"

[providers]
  [providers.file]
    filename = "routes.toml"
```

Create a `routes.toml` file like:

```toml
[http]
  [http.middlewares]
    [http.middlewares.add-foo.addPrefix]
      prefix = "/app_backend"

    [http.middlewares.add-prefix-header.headers]
      customRequestHeaders = { X-Forwarded-Prefix = "/app_backend" }

  [http.routers]
    [http.routers.app-http]
      entryPoints = ["http"]
      service = "app"
      rule = "PathPrefix(`/app_backend`)"
      middlewares = ["add-prefix-header"]

  [http.services]
    [http.services.app]
      [http.services.app.loadBalancer]
        [[http.services.app.loadBalancer.servers]]
          url = "http://127.0.0.1:8080"
```

And run locally with:

`traefik --configFile=traefik.toml`

With the fastapi running now accessing:

http://localhost:9999/app_backend

will take you to a proxy compatible installation
