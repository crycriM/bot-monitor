# Running the Integrated Streamlit Application

## Quick Start

### Step 1: Start the Backend API
```bash
python src/web_api.py --config config/web_processor.yml
```

Expected output:
```
INFO: Uvicorn running on http://0.0.0.0:14440
```

### Step 2: Start the Integrated Streamlit App
```bash
streamlit run src/web_front_sl.py -- --config config/web_processor.yml --gw_port 14440
```

Expected output:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8880
Network URL: http://192.168.x.x:8880
```

### Step 3: Open in Browser
Navigate to: **http://localhost:8880**

## Application Tabs

The integrated Streamlit application now includes 3 main tabs:

### Tab 1: 📊 PnL
- View P&L metrics for different time periods
- Display cumulative and daily performance data
- Filter by strategy

### Tab 2: 🔄 Matching
- View position matching data
- Real vs Theoretical positions comparison
- Mismatch detection and reporting

### Tab 3: 📈 AUM & Performance ⭐ (NEW)
- Cumulative Performance chart (cyan line)
- Cumulative PnL chart (green line)
- Daily Performance chart (green/red bars)
- Data point summary
- Interactive Plotly visualization

## Advanced Usage

### Custom Port
```bash
streamlit run src/web_front_sl.py --server.port 9000 -- --config config/web_processor.yml --gw_port 14440
```

Then access at: http://localhost:9000

### Different Gateway Port
If your backend API is on a different port:
```bash
streamlit run src/web_front_sl.py -- --config config/web_processor.yml --gw_port 15000
```

### Headless Mode (for servers)
```bash
streamlit run src/web_front_sl.py --server.headless true -- --config config/web_processor.yml
```

### Debug Mode
```bash
streamlit run src/web_front_sl.py --logger.level=debug -- --config config/web_processor.yml
```

## Features

### 📊 PnL Tab
- Real-time PnL updates
- Multiple time period views (1d, 7d, 30d, etc.)
- Strategy-level breakdown
- Performance tracking

### 🔄 Matching Tab
- Position comparison (Theo vs Real)
- Mismatch detection
- Account-level matching
- Risk identification

### 📈 AUM & Performance Tab (NEW)
- Real-time performance charts
- Cumulative PnL tracking
- Daily return visualization
- Session/Account selection
- Data statistics
- 60-second cache for performance

## Troubleshooting

### "Unable to connect to backend"
- Verify backend is running on port 14440
- Check firewall settings
- Try: `curl http://localhost:14440/api/available-accounts`

### "No sessions available"
- Backend is running but hasn't loaded data yet
- Wait for backend to initialize
- Check backend logs for errors

### "No graph data available"
- Graph data is only generated when AUM files are processed
- Initial setup may take time
- Wait for 180-day and 30-day metrics to be calculated

### Slow performance
- Data caching is set to 60 seconds
- Multiple users on same instance affects performance
- Consider deploying backend on separate server

### Charts not displaying
- Ensure Plotly is installed: `pip install plotly`
- Check browser console for errors
- Try clearing browser cache

## Performance Optimization

### For Production Deployment

1. **Use dedicated server**
   ```bash
   python src/web_api.py --config config/web_processor.yml &
   streamlit run src/web_front_sl.py -- --config config/web_processor.yml &
   ```

2. **Use reverse proxy (nginx)**
   ```nginx
   location /app {
       proxy_pass http://localhost:8880;
   }
   ```

3. **Enable HTTPS**
   - Configure SSL certificates
   - Use proper reverse proxy setup

4. **Monitor performance**
   - Check Streamlit logs
   - Monitor API response times
   - Track cache hit rates

## Environment Variables

Optional environment variables:

```bash
# Set default gateway port (if API is not on 14440)
export BOT_MONITOR_GATEWAY_PORT=14440

# Set log level
export BOT_MONITOR_LOG_LEVEL=INFO

# Set cache TTL for graphs (seconds)
export BOT_MONITOR_CACHE_TTL=60
```

## Browser Compatibility

Tested and working on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Mobile Responsiveness

The application is responsive and works on:
- ✅ Desktop browsers
- ✅ Tablets
- ⚠️ Mobile (limited charts, better on portrait mode)

## API Health Check

To verify API endpoints are working:

```bash
# Check available accounts
curl http://localhost:14440/api/available-accounts | jq

# Check graph data
curl http://localhost:14440/api/graph-data/binance/binance_1 | jq
```

## Next Steps

1. **Monitor Data**: Use PnL and Matching tabs for real-time monitoring
2. **Track Performance**: Watch AUM & Performance tab for historical trends
3. **Analyze Matches**: Check Matching tab for position discrepancies
4. **Export Data**: Use browser's native export if needed
5. **Set Alerts**: Monitor performance for anomalies

## Support

For issues or questions:
1. Check INTEGRATION_SUMMARY.md for architecture details
2. Review logs in output/ directory
3. Run test_streamlit_implementation.py for verification
4. Check API endpoints with curl commands above

---

## Summary

The integrated Streamlit application provides a complete monitoring solution with:
- ✅ Real-time PnL tracking
- ✅ Position matching
- ✅ Performance visualization
- ✅ Interactive charts
- ✅ Dark theme interface
- ✅ Smart caching
- ✅ API-based architecture

**Ready to use!** 🚀
