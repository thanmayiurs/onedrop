"""
DeliveryOptimizer - Advanced Last Mile Delivery Route Optimization System

Features:
- Enhanced order management and visualization
- Advanced clustering and route optimization
- Cost analysis and statistics
- Interactive maps with detailed routes
- Export capabilities
- Barcode scanning for order entry
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import radians, sin, cos, asin, sqrt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import io
import requests
import time
import os

# Import CV libraries for barcode scanning with error handling
try:
    import cv2
    from pyzbar.pyzbar import decode, ZBarSymbol
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    st.warning("Computer vision libraries not available. Barcode scanning feature will be disabled.")

# Page configuration
st.set_page_config(
    page_title="OneDrop - Last Mile Delivery Optimizer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        margin: 0.5rem 0;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    .stat-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #f0f2f6;
        margin: 0.5rem 0;
        border-left: 4px solid #0066cc;
    }
    .stMetric {
        background-color: rgba(0, 102, 204, 0.1);
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #ffebee;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
EARTH_RADIUS_KM = 6371.0088
FUEL_COST_PER_KM = 12.50
VEHICLE_CAPACITY = 30
DEFAULT_CENTER = {"lat": 12.9716, "lon": 77.5946, "zoom": 11}

# Default hubs
DEFAULT_HUBS = {
    "Koramangala Hub": {"lat": 12.9279, "lon": 77.6271, "address": "Koramangala, Bangalore"},
    "Whitefield Hub": {"lat": 12.9698, "lon": 77.7500, "address": "Whitefield, Bangalore"},
    "Electronic City Hub": {"lat": 12.8399, "lon": 77.6770, "address": "Electronic City, Bangalore"}
}

# Cost constants
DRIVER_COST_PER_DAY = 800.0
MAINTENANCE_COST_PER_KM = 2.50
BASE_DELIVERY_CHARGE = 50.0

def calculate_route_costs(distance_km, num_deliveries):
    """Calculate route costs in INR"""
    fuel_cost = distance_km * FUEL_COST_PER_KM
    maintenance_cost = distance_km * MAINTENANCE_COST_PER_KM
    delivery_charges = num_deliveries * BASE_DELIVERY_CHARGE
    driver_cost = DRIVER_COST_PER_DAY
    
    total_cost = fuel_cost + maintenance_cost + driver_cost
    revenue = delivery_charges
    profit = revenue - total_cost
    
    return {
        'fuel_cost': fuel_cost,
        'maintenance_cost': maintenance_cost,
        'driver_cost': driver_cost,
        'total_cost': total_cost,
        'revenue': revenue,
        'profit': profit
    }

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth"""
    try:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return EARTH_RADIUS_KM * c
    except:
        return 0.0

def get_osrm_route(origin, destination, retry_count=3):
    """Get route using OSRM with error handling"""
    base_url = "http://router.project-osrm.org/route/v1/driving"
    
    for attempt in range(retry_count):
        try:
            url = f"{base_url}/{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
            params = {
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
                "annotations": "false"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data["code"] == "Ok":
                    route = data["routes"][0]
                    coords = [[coord[1], coord[0]] for coord in route["geometry"]["coordinates"]]
                    distance = route["distance"] / 1000
                    return coords, distance
            
            time.sleep(0.5 * (attempt + 1))
            
        except Exception as e:
            st.warning(f"OSRM Error (attempt {attempt+1}/{retry_count}): {str(e)}")
            time.sleep(1)
    
    # Fallback to straight line if all attempts fail
    return [[origin[0], origin[1]], [destination[0], destination[1]]], haversine_km(origin[0], origin[1], destination[0], destination[1])

def generate_sample_orders(n=20):
    """Generate sample delivery orders"""
    rng = np.random.default_rng(42)
    
    areas = [
        ("Koramangala", 12.9279, 77.6271),
        ("Indiranagar", 12.9719, 77.6412),
        ("HSR Layout", 12.9116, 77.6474),
        ("Whitefield", 12.9698, 77.7500),
        ("JP Nagar", 12.9077, 77.5851),
        ("Electronic City", 12.8399, 77.6770),
        ("BTM Layout", 12.9166, 77.6101),
        ("Marathahalli", 12.9591, 77.6974)
    ]
    
    orders = []
    for i in range(n):
        area_name, base_lat, base_lon = areas[i % len(areas)]
        lat = base_lat + float(rng.normal(0, 0.002))
        lon = base_lon + float(rng.normal(0, 0.002))
        
        orders.append({
            'order_id': f'ORD-{i+1:03d}',
            'customer_name': f'Customer {i+1}',
            'area': area_name,
            'lat': lat,
            'lon': lon,
            'packages': int(rng.integers(1, 4))
        })
    
    return pd.DataFrame(orders)

def optimize_routes_with_zones(df, depot, max_vehicles=6):
    """Optimize routes with zone-based distribution"""
    if len(df) == 0:
        return []
    
    coordinates = df[['lat', 'lon']].values
    
    # Calculate number of clusters based on number of orders and vehicles
    n_clusters = min(max(len(df) // 5, 1), max_vehicles)
    
    if len(df) <= n_clusters:
        n_clusters = min(len(df), max_vehicles)
    
    if n_clusters < 1:
        n_clusters = 1
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['zone'] = kmeans.fit_predict(coordinates)
    except Exception as e:
        st.warning(f"Clustering failed: {str(e)}. Using single zone.")
        df['zone'] = 0
    
    routes = []
    assigned_points = set()
    
    # Create routes for each zone
    for zone in range(n_clusters):
        zone_df = df[df['zone'] == zone].copy()
        if len(zone_df) == 0:
            continue
            
        route = []
        current = depot
        unassigned = set(zone_df.index) - assigned_points
        
        # Nearest neighbor approach for route optimization
        while unassigned:
            if len(unassigned) == 0:
                break
                
            distances = []
            for idx in unassigned:
                try:
                    dist = haversine_km(current[0], current[1],
                                      df.loc[idx, 'lat'], 
                                      df.loc[idx, 'lon'])
                    distances.append((idx, dist))
                except:
                    continue
            
            if not distances:
                break
                
            next_point = min(distances, key=lambda x: x[1])[0]
            route.append(next_point)
            assigned_points.add(next_point)
            unassigned.remove(next_point)
            current = (df.loc[next_point, 'lat'], df.loc[next_point, 'lon'])
        
        if route:
            routes.append(route)
    
    # Handle any unassigned points
    remaining_points = set(df.index) - assigned_points
    if remaining_points and len(routes) < max_vehicles:
        remaining_route = list(remaining_points)
        routes.append(remaining_route)
    
    return routes

def create_delivery_map(df, routes=None, depot=None):
    """Create enhanced map with non-overlapping routes"""
    fig = go.Figure()
    
    # Add delivery points
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers',
        marker=dict(
            size=10,
            color='#00ffff',
            opacity=0.8,
            symbol='circle'
        ),
        text=df.apply(lambda x: f"Order: {x['order_id']}<br>Zone: {x.get('zone', 'N/A')}<br>Packages: {x.get('packages', 1)}", axis=1),
        name='Delivery Points',
        hoverinfo='text'
    ))
    
    route_metrics = []
    
    if routes and depot:
        colors = ['#ff1493', '#7fff00', '#4169e1', '#ffd700', '#ff4500', '#9400d3', '#32cd32', '#8a2be2']
        
        # Add depot marker
        fig.add_trace(go.Scattermapbox(
            lat=[depot[0]],
            lon=[depot[1]],
            mode='markers',
            marker=dict(size=20, color='#ff0000', symbol='star'),
            name='Hub/Depot',
            text=f"Hub: {depot[0]:.4f}, {depot[1]:.4f}",
            hoverinfo='text'
        ))
        
        for i, route in enumerate(routes):
            if not route:
                continue
                
            route_points = []
            total_distance = 0
            current = depot
            
            # Get route from depot to first point
            if route:
                first_point = [df.loc[route[0], 'lat'], df.loc[route[0], 'lon']]
                coords, distance = get_osrm_route(current, first_point)
                if coords:
                    route_points.extend(coords)
                    total_distance += distance
                
            # Get routes between delivery points
            for j in range(len(route) - 1):
                current_point = [df.loc[route[j], 'lat'], df.loc[route[j], 'lon']]
                next_point = [df.loc[route[j+1], 'lat'], df.loc[route[j+1], 'lon']]
                coords, distance = get_osrm_route(current_point, next_point)
                
                if coords:
                    route_points.extend(coords)
                    total_distance += distance
            
            # Return to depot from last point
            if route:
                last_point = [df.loc[route[-1], 'lat'], df.loc[route[-1], 'lon']]
                coords, distance = get_osrm_route(last_point, depot)
                if coords:
                    route_points.extend(coords)
                    total_distance += distance
            
            costs = calculate_route_costs(total_distance, len(route))
            route_metrics.append({
                'route': i + 1,
                'distance': total_distance,
                'deliveries': len(route),
                'costs': costs
            })
            
            if route_points:
                lats, lons = zip(*route_points)
                fig.add_trace(go.Scattermapbox(
                    lat=list(lats),
                    lon=list(lons),
                    mode='lines',
                    line=dict(width=3, color=colors[i % len(colors)]),
                    name=f'Route {i+1} ({len(route)} stops, {total_distance:.1f}km)',
                    hovertemplate=(
                        f"<b>Route {i+1}</b><br>"
                        f"Stops: {len(route)}<br>"
                        f"Distance: {total_distance:.1f} km<br>"
                        f"Total Cost: ₹{costs['total_cost']:.0f}<br>"
                        f"Revenue: ₹{costs['revenue']:.0f}<br>"
                        f"Profit: ₹{costs['profit']:.0f}"
                        "<extra></extra>"
                    )
                ))
    
    # Set map center and zoom
    center_lat = df['lat'].mean() if not df.empty else DEFAULT_CENTER["lat"]
    center_lon = df['lon'].mean() if not df.empty else DEFAULT_CENTER["lon"]
    
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=DEFAULT_CENTER["zoom"] if df.empty else 12
        ),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        ),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font=dict(color='white'),
        title=dict(
            text="Delivery Route Optimization Map",
            font=dict(size=20, color='white'),
            x=0.5
        )
    )
    
    return fig, route_metrics

def export_results(df, routes, metrics):
    """Export optimization results"""
    results = {
        'metrics': metrics,
        'routes': [{'route_id': i+1, 'stops': [int(x) for x in r]} for i, r in enumerate(routes)],
        'orders': df.to_dict('records')
    }
    
    json_str = json.dumps(results, indent=2, default=str)
    json_bytes = json_str.encode('utf-8')
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    return json_bytes, csv_buffer.getvalue()

def scan_barcode_from_image(image):
    if not CV_AVAILABLE:
        return None
        
    try:
        # Convert PIL to OpenCV format (RGB → BGR)
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        else:
            image = np.array(image)

        # Convert RGB to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Optional: Apply thresholding for better contrast
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Inside scan_barcode_from_image, after preprocessing:
        if st.checkbox("🔍 Show processed image for debugging"):
            st.image(enhanced, caption="Preprocessed image (grayscale)", clamp=True)
            st.image(thresh, caption="Thresholded image", clamp=True)
        
        # Try decoding on multiple versions
        for img in [enhanced, thresh, gray]:
            barcodes = decode(img)
            if barcodes:
                return barcodes[0].data.decode('utf-8')
                
        return None
    except Exception as e:
        st.error(f"Barcode scanning error: {str(e)}")
        return None

def scan_qr_robust(pil_image):
    """Scan QR code from PIL image. Returns decoded string or None."""
    try:
        img = np.array(pil_image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Convert to grayscale for pyzbar
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try original + enhanced versions
        candidates = [gray]
        
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        candidates.append(clahe.apply(gray))
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(thresh)
        
        for cand in candidates:
            decoded = decode(cand, symbols=[ZBarSymbol.QRCODE])
            if decoded:
                return decoded[0].data.decode('utf-8').strip()
        return None
    except Exception:
        return None

def parse_barcode_data(barcode_data):
    """Parse barcode data into order information"""
    try:
        # Simple parsing - in real app this would depend on barcode format
        data = json.loads(barcode_data)
        return {
            'order_id': data.get('order_id', f'BC-{int(time.time())}'),
            'customer_name': data.get('customer_name', 'Barcode Customer'),
            'area': data.get('area', 'Scanned Area'),
            'lat': float(data.get('lat', DEFAULT_CENTER["lat"])),
            'lon': float(data.get('lon', DEFAULT_CENTER["lon"])),
            'packages': int(data.get('packages', 1))
        }
    except:
        # Fallback parsing for simple text barcodes
        return {
            'order_id': f'BC-{int(time.time())}',
            'customer_name': barcode_data[:20],
            'area': 'Scanned Area',
            'lat': DEFAULT_CENTER["lat"],
            'lon': DEFAULT_CENTER["lon"],
            'packages': 1
        }

def parse_qr_data(qr_str):
    """
    Parse QR code content into order dict.
    Expects JSON: {"order_id":"ORD-001","lat":12.97,"lon":77.59,"customer_name":"John","area":"Koramangala","packages":2}
    Returns dict or None if invalid.
    """
    try:
        import json
        data = json.loads(qr_str)
        
        # Validate required fields
        required = ['order_id', 'lat', 'lon']
        if not all(k in data for k in required):
            return None
            
        return {
            'order_id': str(data['order_id']),
            'customer_name': str(data.get('customer_name', 'QR Customer')),
            'area': str(data.get('area', 'Scanned Area')),
            'lat': float(data['lat']),
            'lon': float(data['lon']),
            'packages': int(data.get('packages', 1))
        }
    except Exception:
        return None

def main():
    st.title("💧 OneDrop Delivery Optimizer")
    st.markdown("### Advanced last-mile delivery route optimization with barcode scanning")
    
    # Initialize session state
    if 'manual_orders' not in st.session_state:
        st.session_state.manual_orders = []
    
    if 'scanned_orders' not in st.session_state:
        st.session_state.scanned_orders = []
    
    # Sidebar controls
    with st.sidebar:
        st.header('🛠️ System Settings')
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Sample Data", "Upload CSV", "Manual Entry", "QR Code Scan"],
            help="Choose how to input delivery orders"
        )
        
        # Vehicle settings
        n_vehicles = st.number_input(
            "Number of Vehicles",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of vehicles available for deliveries"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            vehicle_capacity = st.number_input(
                "Vehicle Capacity (packages)",
                min_value=1,
                value=VEHICLE_CAPACITY,
                help="Maximum packages per vehicle"
            )
            
            fuel_cost = st.number_input(
                "Fuel Cost per KM (₹)",
                min_value=0.0,
                value=FUEL_COST_PER_KM,
                format="%.2f",
                help="Current fuel cost per kilometer"
            )
        
        st.subheader("🏭 Hub Location")
        
        if 'custom_hubs' not in st.session_state:
            st.session_state.custom_hubs = {}
        
        all_hubs = {**DEFAULT_HUBS, **st.session_state.custom_hubs}
        selected_hub = st.selectbox(
            "Select a hub location:",
            list(all_hubs.keys()),
            help="Choose the starting point for all delivery routes"
        )
        hub_info = all_hubs[selected_hub]
        depot_lat, depot_lon = hub_info['lat'], hub_info['lon']
        
        st.info(f"📍 {selected_hub}\n\nCoordinates: {depot_lat:.4f}, {depot_lon:.4f}")
        
        # Add custom hub option
        with st.expander("➕ Add Custom Hub"):
            custom_hub_name = st.text_input("Hub Name")
            custom_hub_lat = st.number_input("Latitude", value=DEFAULT_CENTER["lat"], format="%.6f")
            custom_hub_lon = st.number_input("Longitude", value=DEFAULT_CENTER["lon"], format="%.6f")
            custom_hub_address = st.text_input("Address")
            
            if st.button("Save Custom Hub") and custom_hub_name:
                st.session_state.custom_hubs[custom_hub_name] = {
                    'lat': custom_hub_lat,
                    'lon': custom_hub_lon,
                    'address': custom_hub_address
                }
                st.success(f"Hub '{custom_hub_name}' saved!")
                st.rerun()

    # QR Code scanning section (replaces previous Barcode Scan block)
    if data_source == "QR Code Scan" and CV_AVAILABLE:
        st.subheader("📱 Scan Delivery QR Code")
        
        st.info("""
        📦 **QR Code Format**: Must contain JSON with:
        {
          "order_id": "ORD-001",
          "customer_name": "Rajesh",
          "area": "Koramangala",
          "lat": 12.9279,
          "lon": 77.6271,
          "packages": 2
        }
        Use the QR generator below to create test codes.
        """)
        
        from PIL import Image  # local import to avoid top-level dependency issues
        
        # Upload or camera
        uploaded_img = st.file_uploader("📤 Upload QR image", type=["png", "jpg", "jpeg"])
        camera_img = st.camera_input("📷 Use camera")
        
        image_to_scan = None
        if uploaded_img:
            try:
                image_to_scan = Image.open(uploaded_img)
            except Exception as e:
                st.error(f"❌ Unable to open uploaded image: {e}")
        elif camera_img:
            try:
                image_to_scan = Image.open(camera_img)
            except Exception as e:
                st.error(f"❌ Unable to open camera image: {e}")
        
        if image_to_scan:
            with st.spinner("🔍 Reading QR code..."):
                qr_data = scan_qr_robust(image_to_scan)
                if qr_data:
                    order_data = parse_qr_data(qr_data)
                    if order_data:
                        st.success("✅ QR code scanned successfully!")
                        st.json(order_data)
                        
                        # Add to session state
                        st.session_state.scanned_orders.append(order_data)
                        
                        if st.button("➕ Add to Delivery Orders"):
                            st.session_state.manual_orders.append(order_data)
                            st.success("✅ Order added to list!")
                            st.rerun()
                    else:
                        st.error("❌ Invalid QR format. Must be valid JSON with order_id, lat, lon.")
                else:
                    st.error("❌ No QR code detected. Ensure good lighting and full QR visibility.")
        
        # QR Generator (for testing)
        with st.expander("🛠️ Generate Test QR Code"):
            gen_order_id = st.text_input("Order ID", "QR-001")
            gen_name = st.text_input("Customer Name", "Test Customer")
            gen_lat = st.number_input("Latitude", value=12.9716, format="%.6f")
            gen_lon = st.number_input("Longitude", value=77.5946, format="%.6f")
            gen_area = st.text_input("Area", "Bangalore")
            gen_pkgs = st.number_input("Packages", min_value=1, value=1)
            
            if st.button("🖨️ Generate QR"):
                qr_data_obj = {
                    "order_id": gen_order_id,
                    "customer_name": gen_name,
                    "area": gen_area,
                    "lat": gen_lat,
                    "lon": gen_lon,
                    "packages": gen_pkgs
                }
                qr_str = json.dumps(qr_data_obj)
                
                # Generate QR
                import qrcode
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(qr_str)
                qr.make(fit=True)
                img_qr = qr.make_image(fill='black', back_color='white')
                
                st.image(img_qr, caption="Scan this QR code", width=200)
                st.code(qr_str, language='json')

    elif data_source == "QR Code Scan" and not CV_AVAILABLE:
        st.error("❌ CV libraries not installed. Install: opencv-python pyzbar")  

    # Data loading
    df = None
    
    if data_source == "Sample Data":
        df = generate_sample_orders()
        st.success("✅ Sample data loaded successfully!")
    
    elif data_source == "Upload CSV":
        st.subheader("📤 Upload Delivery Orders")
        uploaded_file = st.file_uploader(
            "📁 Upload CSV file",
            type=['csv'],
            help="CSV should contain: order_id, customer_name, lat, lon, packages (optional)"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # Validate required columns
                required_cols = ['order_id', 'lat', 'lon']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    return
                
                # Fill missing optional columns
                if 'customer_name' not in df.columns:
                    df['customer_name'] = [f'Customer {i+1}' for i in range(len(df))]
                if 'area' not in df.columns:
                    df['area'] = 'Unknown Area'
                if 'packages' not in df.columns:
                    df['packages'] = 1
                
                st.success(f"✅ CSV file uploaded successfully! ({len(df)} orders)")
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                return
        else:
            st.info("Please upload a CSV file to continue")
            return
    
    elif data_source == "Manual Entry":
        st.subheader("📝 Manual Order Entry")
        
        col1, col2 = st.columns(2)
        with col1:
            order_id = st.text_input("Order ID", value=f"ORD-{len(st.session_state.manual_orders)+1:03d}")
            customer_name = st.text_input("Customer Name", value=f"Customer {len(st.session_state.manual_orders)+1}")
            lat = st.number_input("Latitude", value=DEFAULT_CENTER["lat"], format="%.6f")
        with col2:
            area = st.text_input("Area", value="Bangalore Area")
            lon = st.number_input("Longitude", value=DEFAULT_CENTER["lon"], format="%.6f")
            packages = st.number_input("Number of Packages", min_value=1, value=1)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("➕ Add Order"):
                if not order_id or not customer_name:
                    st.error("❌ Order ID and Customer Name are required!")
                else:
                    st.session_state.manual_orders.append({
                        'order_id': order_id,
                        'customer_name': customer_name,
                        'area': area,
                        'lat': lat,
                        'lon': lon,
                        'packages': packages
                    })
                    st.success("✅ Order added successfully!")
                    st.rerun()
        
        with col2:
            if st.session_state.manual_orders and st.button("🗑️ Clear All Orders"):
                st.session_state.manual_orders = []
                st.rerun()
        
        if st.session_state.manual_orders:
            df = pd.DataFrame(st.session_state.manual_orders)
            st.subheader("📋 Current Orders")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Please add at least one order to continue")
            return
    
    # Convert manual orders to DataFrame if needed
    if df is None and st.session_state.manual_orders:
        df = pd.DataFrame(st.session_state.manual_orders)
    
    # Handle empty data
    if df is None or len(df) == 0:
        st.info("No orders available. Please add orders using one of the data sources above.")
        return
    
    # Data validation
    required_cols = ['order_id', 'lat', 'lon']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return
    
    # Show raw data
    with st.expander("📋 View and Edit Data"):
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "lat": st.column_config.NumberColumn("Latitude", format="%.6f"),
                "lon": st.column_config.NumberColumn("Longitude", format="%.6f"),
                "packages": st.column_config.NumberColumn("Packages", min_value=1)
            }
        )
        df = edited_df.copy()
    
    # Optimization section
    st.header("🎯 Route Optimization")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        optimize_button = st.button("🚀 Optimize Routes", type="primary", use_container_width=True)
    
    if optimize_button:
        with st.spinner("🔄 Optimizing delivery routes..."):
            depot = (depot_lat, depot_lon)
            
            try:
                # Optimize routes
                routes = optimize_routes_with_zones(df, depot, max_vehicles=n_vehicles)
                
                if not routes:
                    st.error("❌ No routes could be generated. Please check your data and try again.")
                    return
                
                # Create map and get metrics
                map_fig, route_metrics = create_delivery_map(df, routes, depot)
                
                # Display results
                st.subheader("📊 Optimization Results")
                
                if route_metrics:
                    total_distance = sum(r['distance'] for r in route_metrics)
                    total_deliveries = sum(r['deliveries'] for r in route_metrics)
                    total_cost = sum(r['costs']['total_cost'] for r in route_metrics)
                    total_revenue = sum(r['costs']['revenue'] for r in route_metrics)
                    total_profit = sum(r['costs']['profit'] for r in route_metrics)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Distance", f"{total_distance:.1f} km", 
                                 delta=f"{total_distance/total_deliveries:.1f} km/order")
                    with col2:
                        st.metric("Vehicles Used", len(routes), 
                                 delta=f"{n_vehicles - len(routes)} available")
                    with col3:
                        st.metric("Total Deliveries", total_deliveries)
                    with col4:
                        efficiency = total_deliveries / len(routes) if routes else 0
                        st.metric("Avg Deliveries/Vehicle", f"{efficiency:.1f}")
                    
                    # Cost analysis
                    st.subheader("💰 Cost Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Cost", f"₹{total_cost:.0f}")
                    with col2:
                        st.metric("Total Revenue", f"₹{total_revenue:.0f}")
                    with col3:
                        profit_color = "normal" if total_profit >= 0 else "inverse"
                        st.metric("Net Profit", f"₹{total_profit:.0f}", 
                                 delta=f"{(total_profit/total_cost)*100:.1f}%" if total_cost > 0 else "0%")
                    with col4:
                        cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                        st.metric("Cost per Delivery", f"₹{cost_per_delivery:.0f}")
                    
                    # Route details
                    with st.expander("📋 Route Details"):
                        route_df = pd.DataFrame([{
                            'Route': r['route'],
                            'Deliveries': r['deliveries'],
                            'Distance (km)': f"{r['distance']:.1f}",
                            'Cost (₹)': f"{r['costs']['total_cost']:.0f}",
                            'Revenue (₹)': f"{r['costs']['revenue']:.0f}",
                            'Profit (₹)': f"{r['costs']['profit']:.0f}",
                            'Cost per Delivery': f"₹{r['costs']['total_cost']/r['deliveries']:.0f}" if r['deliveries'] > 0 else "N/A"
                        } for r in route_metrics])
                        st.dataframe(route_df, use_container_width=True, hide_index=True)
                
                # Show map
                st.plotly_chart(map_fig, use_container_width=True)
                
                # Export options
                st.subheader("💾 Export Results")
                
                metrics = {
                    'total_distance': total_distance if route_metrics else 0,
                    'total_cost': total_cost if route_metrics else 0,
                    'total_vehicles': len(routes),
                    'total_deliveries': len(df),
                    'optimization_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                json_data, csv_data = export_results(df, routes, metrics)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.download_button(
                        "📥 Download JSON Results",
                        json_data,
                        file_name=f"delivery_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        "📥 Download Order Data",
                        csv_data,
                        file_name=f"delivery_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col3:
                    # Generate summary report
                    summary = f"""
                    OneDrop Delivery Optimization Report
                    ====================================
                    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    Hub: {selected_hub}
                    Orders Processed: {len(df)}
                    Vehicles Used: {len(routes)}
                    Total Distance: {total_distance:.1f} km
                    Total Cost: ₹{total_cost:.0f}
                    Total Revenue: ₹{total_revenue:.0f}
                    Net Profit: ₹{total_profit:.0f}
                    
                    Route Summary:
                    """
                    for r in route_metrics:
                        summary += f"\nRoute {r['route']}: {r['deliveries']} deliveries, {r['distance']:.1f} km, Cost: ₹{r['costs']['total_cost']:.0f}"
                    
                    st.download_button(
                        "📝 Download Summary",
                        summary,
                        file_name=f"delivery_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col4:
                    if st.button("🔄 Reset Optimization", use_container_width=True):
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
