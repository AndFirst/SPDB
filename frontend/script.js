const API_URL = "http://127.0.0.1:8000/v1/path/";

const MAP_CONFIG = {
  center: [52.2297, 21.0122],
  zoom: 13,
  tileLayer: {
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution:
      '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  },
};

const ICONS = {
  start: L.icon({
    iconUrl:
      "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    shadowSize: [41, 41],
  }),
  default: L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    shadowSize: [41, 41],
  }),
};

const ROUTE_COLORS = {
  defaultRoute: "#3182ce",
  optimizedRoute: "#2ecc71",
};

class Point {
  constructor({ lat, lon, isStart }) {
    this.lat = Number(lat);
    this.lon = Number(lon);
    this.isStart = Boolean(isStart);
    this.validate();
  }

  validate() {
    if (isNaN(this.lat) || isNaN(this.lon)) {
      throw new Error("Invalid Point: lat and lon must be numbers");
    }
    if (this.lat < -90 || this.lat > 90 || this.lon < -180 || this.lon > 180) {
      throw new Error("Invalid Point: lat/lon out of valid range");
    }
  }
}

class RouteStats {
  constructor({ distance, time, numYieldDirections }) {
    this.distance = Number(distance);
    this.time = Number(time);
    this.numYieldDirections = Number(numYieldDirections);
    this.validate();
  }

  validate() {
    if (
      isNaN(this.distance) ||
      isNaN(this.time) ||
      isNaN(this.numYieldDirections) ||
      this.distance < 0 ||
      this.time < 0 ||
      this.numYieldDirections < 0
    ) {
      throw new Error(
        "Invalid RouteStats: distance, time, and numYieldDirections must be non-negative numbers"
      );
    }
  }
}

class RouteData {
  constructor({ path, stats }) {
    this.path = path.map((p) => new Point(p));
    this.stats = new RouteStats(stats);
    this.validate();
  }

  validate() {
    if (!Array.isArray(this.path) || this.path.length === 0) {
      throw new Error("Invalid RouteData: path must be a non-empty array");
    }
  }
}

class ApiResponse {
  constructor(data) {
    this.landmarks = data.landmarks.map((p) => new Point(p));
    this.defaultRoute = new RouteData(data.defaultRoute);
    this.optimizedRoute = new RouteData(data.optimizedRoute);
    this.validate();
  }

  validate() {
    if (!Array.isArray(this.landmarks) || this.landmarks.length === 0) {
      throw new Error(
        "Invalid ApiResponse: landmarks must be a non-empty array"
      );
    }
    const startCount = this.landmarks.filter((p) => p.isStart).length;
    if (startCount !== 1) {
      throw new Error(
        "Invalid ApiResponse: exactly one landmark must be marked as start"
      );
    }
  }
}

class MapState {
  constructor() {
    this.markers = [];
    this.polylines = [];
    this.landmarks = [];
    this.routes = { defaultRoute: null, optimizedRoute: null };
    this.isCalculating = false;
  }

  addPoint({ lat, lon, isStart = false }) {
    this.landmarks.push({ lat, lon, isStart });
  }

  removePoint(index) {
    const wasStart = this.landmarks[index].isStart;
    this.landmarks.splice(index, 1);
    if (wasStart && this.landmarks.length > 0) {
      this.landmarks[0].isStart = true;
    }
  }

  clear() {
    this.landmarks = [];
    this.markers.forEach((marker) => marker.remove());
    this.markers = [];
    this.polylines.forEach((polyline) => polyline.remove());
    this.polylines = [];
    this.routes = { defaultRoute: null, optimizedRoute: null };
    this.isCalculating = false;
  }

  setStartPoint(index) {
    this.landmarks.forEach((point) => (point.isStart = false));
    this.landmarks[index].isStart = true;
  }

  setRouteData(routeType, data) {
    this.routes[routeType] = data;
  }
}

function initializeMap() {
  const map = L.map("map").setView(MAP_CONFIG.center, MAP_CONFIG.zoom);
  L.tileLayer(MAP_CONFIG.tileLayer.url, {
    attribution: MAP_CONFIG.tileLayer.attribution,
  }).addTo(map);
  return map;
}

const UIManager = {
  pointsList: document.getElementById("points"),
  statistics: document.getElementById("statistics"),

  updatePointsList(state) {
    this.pointsList.innerHTML = "";
    state.landmarks.forEach((point, index) => {
      const li = document.createElement("li");
      li.className = `point-item ${point.isStart ? "start-point" : ""}`;
      li.innerHTML = `Punkt ${index + 1}: (${point.lat.toFixed(
        4
      )}, ${point.lon.toFixed(4)})${
        point.isStart ? ' <span class="start-indicator">➤</span>' : ""
      } <span class="delete-btn" data-index="${index}">🗑</span>`;
      li.replaceWith(li.cloneNode(true));
      li.querySelector(".delete-btn").addEventListener("click", () => {
        state.removePoint(index);
        this.updatePointsList(state);
        updateMarkers(state, map);
        if (state.routes.defaultRoute || state.routes.optimizedRoute) {
          state.routes = { defaultRoute: null, optimizedRoute: null };
          updateRouteDisplay(state, map);
          this.toggleStatistics(false);
        }
      });
      li.addEventListener("click", (e) => {
        if (e.target.className !== "delete-btn") {
          state.setStartPoint(index);
          this.updatePointsList(state);
          updateMarkers(state, map);
        }
      });
      this.pointsList.appendChild(li);
    });
  },

  toggleStatistics(show) {
    this.statistics.style.display = show ? "block" : "none";
  },

  updateStatistics(state) {
    const defaultStats = state.routes.defaultRoute?.stats || {
      time: 0,
      distance: 0,
      numYieldDirections: 0,
    };
    const optimizedStats = state.routes.optimizedRoute?.stats || {
      time: 0,
      distance: 0,
      numYieldDirections: 0,
    };

    document.getElementById("stat-time").innerText =
      defaultStats.time.toFixed(2);
    document.getElementById("stat-distance").innerText =
      defaultStats.distance.toFixed(2);
    document.getElementById("stat-left-turns").innerText =
      defaultStats.numYieldDirections;

    document.getElementById("stat-time-left-turns").innerText =
      optimizedStats.time.toFixed(2);
    document.getElementById("stat-distance-left-turns").innerText =
      optimizedStats.distance.toFixed(2);
    document.getElementById("stat-left-turns-left-turns").innerText =
      optimizedStats.numYieldDirections;

    this.toggleStatistics(true);
  },
};

function updateMarkers(state, map) {
  state.markers.forEach((marker) => marker.remove());
  state.markers = [];

  state.landmarks.forEach((point, index) => {
    const marker = L.marker([point.lat, point.lon], {
      icon: point.isStart ? ICONS.start : ICONS.default,
    })
      .addTo(map)
      .bindPopup(`Punkt ${index + 1}${point.isStart ? " (Początek)" : ""}`);
    state.markers.push(marker);
  });
}

async function _calculateRoute(state, map) {
  if (state.isCalculating) {
    console.log("calculateRoute: Already calculating, skipping...");
    return;
  }

  if (state.landmarks.length < 2) {
    alert("Proszę wybrać co najmniej 2 punkty!");
    return;
  }

  const startPoints = state.landmarks.filter((point) => point.isStart).length;
  if (startPoints !== 1) {
    alert("Musi być dokładnie jeden punkt początkowy!");
    return;
  }

  state.isCalculating = true;

  try {
    state.polylines.forEach((polyline) => polyline.remove());
    state.polylines = [];

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks: state.landmarks }),
    });

    if (!response.ok) throw new Error("API Error: " + response.statusText);
    const data = await response.json();
    const validatedData = new ApiResponse(data);

    state.setRouteData("defaultRoute", validatedData.defaultRoute);
    state.setRouteData("optimizedRoute", validatedData.optimizedRoute);

    updateRouteDisplay(state, map);
    UIManager.updateStatistics(state);
  } catch (error) {
    console.error("Błąd:", error);
    alert("Wystąpił błąd podczas obliczania trasy: " + error.message);
  } finally {
    state.isCalculating = false;
  }
}

function updateRouteDisplay(state, map) {
  updateMarkers(state, map);

  state.polylines.forEach((polyline) => polyline.remove());
  state.polylines = [];

  const bounds = L.latLngBounds();

  if (state.routes.defaultRoute) {
    const routeCoords = state.routes.defaultRoute.path.map((coord) => [
      coord.lat,
      coord.lon,
    ]);
    const polyline = L.polyline(routeCoords, {
      color: ROUTE_COLORS.defaultRoute,
      weight: 7,
      opacity: 0.8,
      zIndex: 1,
    }).addTo(map);
    state.polylines.push(polyline);
    bounds.extend(polyline.getBounds());

    polyline.on("mouseover", function () {
      this.setStyle({ weight: 10, opacity: 1 });
    });
    polyline.on("mouseout", function () {
      this.setStyle({ weight: 7, opacity: 0.8 });
    });
  }

  if (state.routes.optimizedRoute) {
    const routeCoords = state.routes.optimizedRoute.path.map((coord) => [
      coord.lat,
      coord.lon,
    ]);
    const polyline = L.polyline(routeCoords, {
      color: ROUTE_COLORS.optimizedRoute,
      weight: 5,
      opacity: 0.8,
      dashArray: "10, 10",
      zIndex: 2,
    }).addTo(map);
    state.polylines.push(polyline);
    bounds.extend(polyline.getBounds());

    polyline.on("mouseover", function () {
      this.setStyle({ weight: 8, opacity: 1, dashArray: null });
    });
    polyline.on("mouseout", function () {
      this.setStyle({ weight: 5, opacity: 0.8, dashArray: "10, 10" });
    });
  }

  map.on("click", () => {
    state.polylines.forEach((polyline) => {
      if (polyline.options.color === ROUTE_COLORS.defaultRoute) {
        polyline.setStyle({ weight: 7, opacity: 0.8 });
      } else {
        polyline.setStyle({ weight: 5, opacity: 0.8, dashArray: "10, 10" });
      }
    });
  });

  if (bounds.isValid()) {
    map.fitBounds(bounds);
  }
}

const map = initializeMap();
const state = new MapState();
UIManager.toggleStatistics(false);

map.on("click", (e) => {
  const { lat, lng: lon } = e.latlng;
  const isStart = state.landmarks.length === 0;

  state.addPoint({ lat, lon, isStart });
  updateMarkers(state, map);
  UIManager.updatePointsList(state);
});

function clearAll() {
  state.clear();
  UIManager.updatePointsList(state);
  UIManager.toggleStatistics(false);
}

window.calculateRoute = () => _calculateRoute(state, map);
window.clearAll = clearAll;
