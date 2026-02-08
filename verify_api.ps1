$baseUrl = "http://localhost:8000"

Write-Host "Testing Root Endpoint..."
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/" -Method Get
    Write-Host "Success: $($response.message)" -ForegroundColor Green
} catch {
    Write-Host "Failed to connect to root endpoint: $_" -ForegroundColor Red
}

Write-Host "`nTesting Wellness Analysis..."
$wellnessBody = @{
    steps = 8500
    heart_rate = 72
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/analyze/wellness" -Method Post -Body $wellnessBody -ContentType "application/json"
    Write-Host "Success! Advice: $($response.advice)" -ForegroundColor Green
} catch {
    Write-Host "Failed wellness analysis: $_" -ForegroundColor Red
}

Write-Host "`nTesting Face Analysis (Mock)..."
$faceBody = @{
    image = "base64_mock_string"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/analyze/face" -Method Post -Body $faceBody -ContentType "application/json"
    Write-Host "Success! Happiness: $($response.happiness)" -ForegroundColor Green
} catch {
    Write-Host "Failed face analysis: $_" -ForegroundColor Red
}
