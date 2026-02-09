# Stage 1: Build the React app
FROM node:20-slim as build-stage

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Serve the app with Nginx
FROM nginx:alpine

# Copy the custom nginx config
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy build artifacts from build-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
