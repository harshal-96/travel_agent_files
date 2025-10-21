"use client"

import { useEffect, useState, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Map, Loader2, MapPin, Hotel, Utensils, Camera, Navigation, X, Star, DollarSign } from "lucide-react"

interface Location {
  name: string
  address: string
  rating: number
  reviews: number
  lat: number
  lng: number
  types: string[]
  place_id: string
  type?: string
}

interface TravelPlanData {
  destination: string
  origin?: string
  locations?: {
    hotels?: Location[]
    restaurants?: Location[]
    attractions?: Location[]
    specific_places?: Location[]
  }
}

interface TravelMapViewProps {
  travelPlanData: TravelPlanData
  searchResults?: any
}

export function TravelMapView({ travelPlanData, searchResults }: TravelMapViewProps) {
  const [mapLoaded, setMapLoaded] = useState(false)
  const [apiKey, setApiKey] = useState<string | null>(null)
  const [selectedPlace, setSelectedPlace] = useState<Location | null>(null)
  const [mapInstance, setMapInstance] = useState<any>(null)
  const [markers, setMarkers] = useState<any[]>([])
  const [filter, setFilter] = useState<string>("all")
  const mapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchGoogleMapsApiKey()
  }, [])

  const fetchGoogleMapsApiKey = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/maps-api-key")
      if (response.ok) {
        const data = await response.json()
        setApiKey(data.apiKey)
      }
    } catch (error) {
      console.error("Error fetching Google Maps API key:", error)
    }
  }

  useEffect(() => {
    if (apiKey && !mapLoaded) {
      initializeMap()
      setMapLoaded(true)
    }
  }, [apiKey, mapLoaded])

  useEffect(() => {
    if (mapInstance && travelPlanData) {
      updateMapMarkers()
    }
  }, [mapInstance, travelPlanData, filter])

  const initializeMap = () => {
    if ((window as any).google) {
      renderMap()
      return
    }

    const script = document.createElement("script")
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places`
    script.async = true
    script.onload = renderMap
    document.head.appendChild(script)
  }

  const renderMap = () => {
    const destination = travelPlanData?.destination || ""

    if ((window as any).google && destination && mapRef.current) {
      const map = new (window as any).google.maps.Map(mapRef.current, {
        center: { lat: 0, lng: 0 },
        zoom: 13,
        styles: [
          {
            featureType: "poi",
            elementType: "labels",
            stylers: [{ visibility: "off" }]
          }
        ],
        mapTypeControl: true,
        fullscreenControl: true,
        streetViewControl: true,
        zoomControl: true
      })

      setMapInstance(map)

      const geocoder = new (window as any).google.maps.Geocoder()
      geocoder.geocode({ address: destination }, (results: any, status: any) => {
        if (status === "OK" && results && results[0]) {
          const centerLocation = results[0].geometry.location
          map.setCenter(centerLocation)
        }
      })
    }
  }

  const updateMapMarkers = () => {
    if (!mapInstance || !travelPlanData?.locations) return

    // Clear existing markers
    markers.forEach(marker => marker.setMap(null))
    const newMarkers: any[] = []

    const locations = travelPlanData.locations
    const bounds = new (window as any).google.maps.LatLngBounds()

    // Helper function to get SVG icon for category
    const getSVGIcon = (category: string) => {
      const icons = {
        hotel: `
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1E40AF" stroke="white" stroke-width="1">
            <path d="M7 13c1.66 0 3-1.34 3-3S8.66 7 7 7s-3 1.34-3 3 1.34 3 3 3zm12-6h-8v7H3V6H1v15h2v-3h18v3h2v-9c0-2.21-1.79-4-4-4z"/>
          </svg>
        `,
        restaurant: `
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#D97706" stroke="white" stroke-width="1">
            <path d="M11 9H9V2H7v7H5V2H3v7c0 2.12 1.66 3.84 3.75 3.97V22h2.5v-9.03C11.34 12.84 13 11.12 13 9V2h-2v7zm5-3v8h2.5v8H21V2c-2.76 0-5 2.24-5 4z"/>
          </svg>
        `,
        attraction: `
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1D4ED8" stroke="white" stroke-width="1">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
          </svg>
        `,
        place: `
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#7C3AED" stroke="white" stroke-width="1">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
          </svg>
        `
      }
      return icons[category as keyof typeof icons] || icons.place
    }

    // Helper function to create markers
    const createMarker = (location: Location, icon: string, color: string, category: string) => {
      if (!location.lat || !location.lng) return

      const position = { lat: location.lat, lng: location.lng }
      
      // Create custom SVG icon
      const svgIcon = getSVGIcon(category)
      const iconUrl = 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svgIcon)
      
      const marker = new (window as any).google.maps.Marker({
        map: mapInstance,
        position: position,
        title: location.name,
        icon: {
          url: iconUrl,
          scaledSize: new (window as any).google.maps.Size(32, 32),
          anchor: new (window as any).google.maps.Point(16, 32)
        },
        category: category,
        locationData: location,
        animation: (window as any).google.maps.Animation.DROP
      })

      bounds.extend(position)

      const infoWindow = new (window as any).google.maps.InfoWindow({
        content: createInfoWindowContent(location, category)
      })

      marker.addListener("click", () => {
        setSelectedPlace(location)
        infoWindow.open(mapInstance, marker)
      })

      return marker
    }

    // Add specific places (highest priority)
    if (locations.specific_places && (filter === "all" || filter === "attractions")) {
      locations.specific_places.forEach((place: Location) => {
        const marker = createMarker(place, "attractions", "#8B5CF6", "attraction")
        if (marker) newMarkers.push(marker)
      })
    }

    // Add hotels
    if (locations.hotels && (filter === "all" || filter === "hotels")) {
      locations.hotels.forEach((hotel: Location) => {
        const marker = createMarker(hotel, "hotel", "#EF4444", "hotel")
        if (marker) newMarkers.push(marker)
      })
    }

    // Add restaurants
    if (locations.restaurants && (filter === "all" || filter === "restaurants")) {
      locations.restaurants.forEach((restaurant: Location) => {
        const marker = createMarker(restaurant, "restaurant", "#F59E0B", "restaurant")
        if (marker) newMarkers.push(marker)
      })
    }

    // Add general attractions
    if (locations.attractions && (filter === "all" || filter === "attractions")) {
      locations.attractions.forEach((attraction: Location) => {
        const marker = createMarker(attraction, "attraction", "#3B82F6", "attraction")
        if (marker) newMarkers.push(marker)
      })
    }

    setMarkers(newMarkers)

    // Fit map to show all markers
    if (newMarkers.length > 0) {
      mapInstance.fitBounds(bounds)
    }
  }

  const createInfoWindowContent = (location: Location, category: string) => {
    const rating = location.rating || 0
    const reviews = location.reviews || 0
    
    return `
      <div style="padding: 12px; max-width: 250px;">
        <h3 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600; color: #1f2937;">
          ${location.name}
        </h3>
        <div style="display: flex; align-items: center; gap: 4px; margin-bottom: 8px;">
          <span style="color: #f59e0b; font-size: 14px;">⭐</span>
          <span style="font-size: 14px; font-weight: 500;">${rating.toFixed(1)}</span>
          <span style="font-size: 12px; color: #6b7280;">(${reviews} reviews)</span>
        </div>
        <p style="margin: 0 0 8px 0; font-size: 13px; color: #4b5563;">
          ${location.address}
        </p>
        <div style="display: inline-block; padding: 4px 8px; background: #dbeafe; color: #1e40af; border-radius: 4px; font-size: 11px; font-weight: 500; text-transform: capitalize;">
          ${category}
        </div>
      </div>
    `
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case "hotels":
        return <Hotel className="w-4 h-4" />
      case "restaurants":
        return <Utensils className="w-4 h-4" />
      case "attractions":
        return <Camera className="w-4 h-4" />
      default:
        return <MapPin className="w-4 h-4" />
    }
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "hotels":
        return "bg-red-500"
      case "restaurants":
        return "bg-amber-500"
      case "attractions":
        return "bg-blue-500"
      default:
        return "bg-purple-500"
    }
  }

  const getCategoryCount = (category: string) => {
    if (!travelPlanData?.locations) return 0
    
    switch (category) {
      case "hotels":
        return travelPlanData.locations.hotels?.length || 0
      case "restaurants":
        return travelPlanData.locations.restaurants?.length || 0
      case "attractions":
        return (travelPlanData.locations.attractions?.length || 0) + 
               (travelPlanData.locations.specific_places?.length || 0)
      default:
        return 0
    }
  }

  const totalLocations = 
    (travelPlanData?.locations?.hotels?.length || 0) +
    (travelPlanData?.locations?.restaurants?.length || 0) +
    (travelPlanData?.locations?.attractions?.length || 0) +
    (travelPlanData?.locations?.specific_places?.length || 0)

  return (
    <Card className="shadow-lg h-full flex flex-col">
      <CardHeader className="pb-3 border-b">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Map className="w-5 h-5 text-green-600" />
            Interactive Map
            <span className="text-xs font-normal text-gray-500">
              ({totalLocations} locations)
            </span>
          </CardTitle>
          <Navigation className="w-4 h-4 text-gray-400" />
        </div>
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col p-4 gap-3">
        {!apiKey ? (
          <div className="flex-1 flex items-center justify-center bg-slate-100 rounded-lg">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-2" />
              <p className="text-sm text-gray-600">Loading map...</p>
            </div>
          </div>
        ) : (
          <>
            {/* Filter Buttons */}
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => setFilter("all")}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                  filter === "all"
                    ? "bg-gray-900 text-white"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                }`}
              >
                All ({totalLocations})
              </button>
              <button
                onClick={() => setFilter("hotels")}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1 ${
                  filter === "hotels"
                    ? "bg-blue-700 text-white"
                    : "bg-blue-50 text-blue-800 hover:bg-blue-100"
                }`}
              >
                <Hotel className="w-3 h-3" />
                Hotels ({getCategoryCount("hotels")})
              </button>
              <button
                onClick={() => setFilter("restaurants")}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1 ${
                  filter === "restaurants"
                    ? "bg-amber-600 text-white"
                    : "bg-amber-50 text-amber-800 hover:bg-amber-100"
                }`}
              >
                <Utensils className="w-3 h-3" />
                Dining ({getCategoryCount("restaurants")})
              </button>
              <button
                onClick={() => setFilter("attractions")}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1 ${
                  filter === "attractions"
                    ? "bg-blue-600 text-white"
                    : "bg-blue-50 text-blue-800 hover:bg-blue-100"
                }`}
              >
                <Camera className="w-3 h-3" />
                Attractions ({getCategoryCount("attractions")})
              </button>
            </div>

            {/* Map Container */}
            <div 
              ref={mapRef}
              className="w-full flex-1 bg-slate-200 rounded-lg shadow-inner min-h-[400px]"
            />

            {/* Selected Place Details */}
            {selectedPlace && (
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-3 border border-blue-200">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-white rounded-lg shadow-sm">
                      {getCategoryIcon(selectedPlace.type || "place")}
                    </div>
                    <div>
                      <h4 className="font-semibold text-sm text-gray-900">
                        {selectedPlace.name}
                      </h4>
                      <div className="flex items-center gap-1 mt-0.5">
                        <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                        <span className="text-xs font-medium text-gray-700">
                          {selectedPlace.rating?.toFixed(1) || "N/A"}
                        </span>
                        <span className="text-xs text-gray-500">
                          ({selectedPlace.reviews || 0})
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedPlace(null)}
                    className="p-1 hover:bg-white rounded-full transition-colors"
                  >
                    <X className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
                <p className="text-xs text-gray-600 line-clamp-2">
                  {selectedPlace.address}
                </p>
              </div>
            )}

            {/* Legend */}
            <div className="grid grid-cols-2 gap-2 pt-2 border-t">
              <div className="flex items-center gap-2">
                <Hotel className="w-5 h-5 text-blue-800" />
                <span className="text-xs text-gray-700">Hotels</span>
              </div>
              <div className="flex items-center gap-2">
                <Utensils className="w-5 h-5 text-amber-600" />
                <span className="text-xs text-gray-700">Restaurants</span>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="w-5 h-5 text-red-600" />
                <span className="text-xs text-gray-700">Attractions</span>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="w-5 h-5 text-purple-600" />
                <span className="text-xs text-gray-700">Featured</span>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}