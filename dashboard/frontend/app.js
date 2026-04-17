// Trading AI System Dashboard - Frontend Application

// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

// Authentication Token (would be set after login)
let authToken = localStorage.getItem('authToken') || '';

// Global State
let state = {
    connected: false,
    mode: 'paper', // paper or live
    currentPage: 'dashboard',
    charts: {},
    updateInterval: null,
    socket: null,
    marketData: {
        bid: 0,
        ask: 0,
        spread: 0
    },
    positions: [],
    trades: [],
    signals: [],
    accountInfo: {
        balance: 10000,
        equity: 10000,
        dailyPnL: 0
    }
};

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    setupWebSocket();
    startDataUpdates();
    loadInitialData();
});

// Initialize Application
function initializeApp() {
    // Setup navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            switchPage(page);
        });
    });
    
    // Initialize charts
    initializeCharts();
    
    // Check authentication
    if (!authToken) {
        showToast('Please login to continue', 'warning');
        // Show login modal (simplified - would implement proper auth)
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Mode toggle
    document.getElementById('modeToggle')?.addEventListener('click', toggleMode);
    
    // Trading buttons
    document.getElementById('buyBtn')?.addEventListener('click', () => placeOrder('BUY'));
    document.getElementById('sellBtn')?.addEventListener('click', () => placeOrder('SELL'));
    document.getElementById('submitOrder')?.addEventListener('click', () => {
        const activeBtn = document.querySelector('.dir-btn.active');
        if (activeBtn) {
            placeOrder(activeBtn.classList.contains('buy-btn') ? 'BUY' : 'SELL');
        }
    });
    
    // Close all positions
    document.getElementById('closeAllPositions')?.addEventListener('click', closeAllPositions);
    
    // Refresh buttons
    document.getElementById('refreshSignals')?.addEventListener('click', loadSignals);
    document.getElementById('refreshHistory')?.addEventListener('click', loadTradeHistory);
    
    // Export
    document.getElementById('exportHistory')?.addEventListener('click', exportTradeHistory);
    
    // Kill switch
    document.getElementById('triggerKillSwitch')?.addEventListener('click', triggerKillSwitch);
    
    // Settings
    document.getElementById('saveSettings')?.addEventListener('click', saveSettings);
    document.getElementById('resetSettings')?.addEventListener('click', resetSettings);
    
    // Chart controls
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const period = e.target.dataset.chart;
            updateEquityChart(period);
        });
    });
    
    document.querySelectorAll('.tool-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const interval = e.target.dataset.interval;
            changeChartInterval(interval);
        });
    });
    
    // Indicator toggles
    document.getElementById('showEMA20')?.addEventListener('change', toggleIndicator);
    document.getElementById('showEMA50')?.addEventListener('change', toggleIndicator);
    document.getElementById('showSMCLevels')?.addEventListener('change', toggleIndicator);
    
    // Order form inputs
    document.getElementById('lotSize')?.addEventListener('input', updateRiskInfo);
    document.getElementById('stopLoss')?.addEventListener('input', updateRiskInfo);
    document.getElementById('takeProfit')?.addEventListener('input', updateRiskInfo);
}

// Initialize Charts
function initializeCharts() {
    // Equity Chart
    const equityCtx = document.getElementById('equityChart')?.getContext('2d');
    if (equityCtx) {
        state.charts.equity = new Chart(equityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity',
                    data: [],
                    borderColor: '#0D6EFD',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#F8FAFC' } }
                },
                scales: {
                    x: { grid: { color: '#334155' }, ticks: { color: '#94A3B8' } },
                    y: { grid: { color: '#334155' }, ticks: { color: '#94A3B8' } }
                }
            }
        });
    }
    
    // P&L Distribution Chart
    const pnlCtx = document.getElementById('pnlDistributionChart')?.getContext('2d');
    if (pnlCtx) {
        state.charts.pnlDistribution = new Chart(pnlCtx, {
            type: 'bar',
            data: {
                labels: ['-100', '-50', '0', '+50', '+100', '+150', '+200'],
                datasets: [{
                    label: 'Trade Count',
                    data: [0, 0, 0, 0, 0, 0, 0],
                    backgroundColor: '#0D6EFD'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#F8FAFC' } }
                },
                scales: {
                    x: { grid: { color: '#334155' }, ticks: { color: '#94A3B8' } },
                    y: { grid: { color: '#334155' }, ticks: { color: '#94A3B8' } }
                }
            }
        });
    }
    
    // Confidence Distribution Chart
    const confCtx = document.getElementById('confidenceChart')?.getContext('2d');
    if (confCtx) {
        state.charts.confidence = new Chart(confCtx, {
            type: 'doughnut',
            data: {
                labels: ['High (>80%)', 'Medium (65-80%)', 'Low (<65%)'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#00FF88', '#FFB800', '#FF4444']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#F8FAFC' } }
                }
            }
        });
    }
    
    // Feature Importance Chart
    const featureCtx = document.getElementById('featureImportanceChart')?.getContext('2d');
    if (featureCtx) {
        state.charts.featureImportance = new Chart(featureCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'SHAP Value',
                    data: [],
                    backgroundColor: '#0D6EFD'
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { color: '#334155' }, ticks: { color: '#94A3B8' } },
                    y: { ticks: { color: '#94A3B8' } }
                }
            }
        });
    }
    
    // Live Chart (Trading Page)
    const liveCtx = document.getElementById('liveChart')?.getContext('2d');
    if (liveCtx) {
        state.charts.live = new Chart(liveCtx, {
            type: 'candlestick',
            data: { datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#F8FAFC' } }
                }
            }
        });
    }
    
    // Exposure Chart
    const exposureCtx = document.getElementById('exposureChart')?.getContext('2d');
    if (exposureCtx) {
        state.charts.exposure = new Chart(exposureCtx, {
            type: 'pie',
            data: {
                labels: ['Long', 'Short', 'Cash'],
                datasets: [{
                    data: [0, 0, 100],
                    backgroundColor: ['#00FF88', '#FF4444', '#6C757D']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#F8FAFC' } }
                }
            }
        });
    }
    
    // Drift Chart
    const driftCtx = document.getElementById('driftChart')?.getContext('2d');
    if (driftCtx) {
        state.charts.drift = new Chart(driftCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'PSI Score',
                    data: [],
                    borderColor: '#FFB800',
                    fill: false
                }, {
                    label: 'Confidence',
                    data: [],
                    borderColor: '#00FF88',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#F8FAFC' } }
                },
                scales: {
                    x: { grid: { color: '#334155' }, ticks: { color: '#94A3B8' } },
                    y: { grid: { color: '#334155' }, ticks: { color: '#94A3B8' } }
                }
            }
        });
    }
}

// Switch Page
function switchPage(pageName) {
    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.page === pageName) {
            item.classList.add('active');
        }
    });
    
    // Update visible page
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(`${pageName}Page`).classList.add('active');
    
    // Update header title
    const titles = {
        dashboard: 'Dashboard',
        trading: 'Trading Terminal',
        positions: 'Positions',
        history: 'Trade History',
        models: 'Model Management',
        risk: 'Risk Management',
        settings: 'Settings'
    };
    document.getElementById('pageTitle').textContent = titles[pageName] || 'Dashboard';
    
    state.currentPage = pageName;
    
    // Load page-specific data
    if (pageName === 'positions') loadPositions();
    if (pageName === 'history') loadTradeHistory();
    if (pageName === 'models') loadModels();
}

// Setup WebSocket for real-time data
function setupWebSocket() {
    state.socket = io(WS_URL, {
        transports: ['websocket'],
        query: { token: authToken }
    });
    
    state.socket.on('connect', () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    });
    
    state.socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
    });
    
    state.socket.on('market_data', (data) => {
        updateMarketData(data);
    });
    
    state.socket.on('signal', (signal) => {
        updateSignalDisplay(signal);
        addSignalToTable(signal);
        showToast(`New AI Signal: ${signal.direction} (${(signal.confidence * 100).toFixed(1)}%)`, 'info');
    });
    
    state.socket.on('trade_update', (update) => {
        loadPositions();
        loadTradeHistory();
        updateAccountInfo();
    });
    
    state.socket.on('alert', (alert) => {
        showToast(alert.message, alert.type || 'warning');
        if (alert.type === 'danger') {
            updateKillSwitchStatus(true);
        }
    });
}

// Start Periodic Data Updates
function startDataUpdates() {
    state.updateInterval = setInterval(() => {
        updateAccountInfo();
        if (state.currentPage === 'positions') loadPositions();
        if (state.currentPage === 'dashboard') loadSignals();
    }, 5000);
}

// Load Initial Data
async function loadInitialData() {
    await Promise.all([
        updateAccountInfo(),
        loadSignals(),
        loadPositions(),
        loadTradeHistory(),
        loadModels()
    ]);
}

// API Calls
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json'
        }
    };
    
    if (body) options.body = JSON.stringify(body);
    
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showToast(`API Error: ${error.message}`, 'error');
        return null;
    }
}

// Update Account Information
async function updateAccountInfo() {
    const account = await apiCall('/account');
    if (account) {
        state.accountInfo = account;
        
        document.getElementById('balance').textContent = `$${account.balance.toFixed(2)}`;
        document.getElementById('equity').textContent = `$${account.equity.toFixed(2)}`;
        
        const dailyPnL = account.realized_pnl_today || 0;
        document.getElementById('dailyPnL').textContent = `${dailyPnL >= 0 ? '+' : ''}$${dailyPnL.toFixed(2)}`;
        
        const dailyPercent = (dailyPnL / account.balance) * 100;
        document.getElementById('dailyPnLPercent').textContent = `${dailyPercent >= 0 ? '+' : ''}${dailyPercent.toFixed(2)}%`;
        document.getElementById('dailyPnLPercent').className = `stat-change ${dailyPercent >= 0 ? 'positive' : 'negative'}`;
    }
}

// Load Signals
async function loadSignals() {
    const signals = await apiCall('/signals?limit=50');
    if (signals) {
        state.signals = signals;
        renderSignalsTable(signals);
        
        // Update win rate
        const closedTrades = signals.filter(s => s.status === 'closed');
        if (closedTrades.length > 0) {
            const wins = closedTrades.filter(t => t.pnl > 0).length;
            const winRate = (wins / closedTrades.length) * 100;
            document.getElementById('winRate').textContent = `${winRate.toFixed(1)}%`;
        }
    }
}

// Load Positions
async function loadPositions() {
    const positions = await apiCall('/positions');
    if (positions) {
        state.positions = positions;
        renderPositionsTable(positions);
        
        document.getElementById('openPositionsCount').textContent = positions.length;
        
        const totalExposure = positions.reduce((sum, p) => sum + (p.quantity * p.open_price), 0);
        document.getElementById('totalExposure').textContent = `$${totalExposure.toFixed(2)}`;
        
        const unrealizedPnL = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);
        document.getElementById('unrealizedPnL').textContent = `${unrealizedPnL >= 0 ? '+' : ''}$${unrealizedPnL.toFixed(2)}`;
        document.getElementById('unrealizedPnL').className = `big-number ${unrealizedPnL >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
    }
}

// Load Trade History
async function loadTradeHistory() {
    const startDate = document.getElementById('historyStartDate')?.value;
    const endDate = document.getElementById('historyEndDate')?.value;
    const symbol = document.getElementById('historySymbol')?.value;
    
    let url = '/trades/history';
    const params = new URLSearchParams();
    if (startDate) params.append('start', startDate);
    if (endDate) params.append('end', endDate);
    if (symbol) params.append('symbol', symbol);
    if (params.toString()) url += `?${params.toString()}`;
    
    const trades = await apiCall(url);
    if (trades) {
        state.trades = trades;
        renderHistoryTable(trades);
        document.getElementById('tradeCount').textContent = `${trades.length} trades`;
        
        // Update profit factor
        const grossProfit = trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0);
        const grossLoss = Math.abs(trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));
        const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;
        document.getElementById('profitFactor').textContent = profitFactor.toFixed(2);
        
        // Update P&L distribution chart
        updatePnLDistribution(trades);
    }
}

// Load Models
async function loadModels() {
    const models = await apiCall('/models');
    if (models) {
        // Update feature importance chart
        if (models.feature_importance) {
            const features = Object.entries(models.feature_importance)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10);
            
            if (state.charts.featureImportance) {
                state.charts.featureImportance.data.labels = features.map(f => f[0]);
                state.charts.featureImportance.data.datasets[0].data = features.map(f => f[1]);
                state.charts.featureImportance.update();
            }
        }
        
        // Update drift metrics
        if (models.drift) {
            document.getElementById('psiScore').textContent = models.drift.psi_score.toFixed(3);
            document.getElementById('predConfidence').textContent = `${(models.drift.avg_confidence * 100).toFixed(1)}%`;
            document.getElementById('lastRetrain').textContent = models.drift.last_retrain || 'Never';
            
            const psiStatus = models.drift.psi_score < 0.1 ? 'stable' : (models.drift.psi_score < 0.2 ? 'warning' : 'critical');
            document.getElementById('psiStatus').className = `drift-status ${psiStatus}`;
            document.getElementById('psiStatus').textContent = psiStatus.charAt(0).toUpperCase() + psiStatus.slice(1);
        }
    }
}

// Place Order
async function placeOrder(direction) {
    const lotSize = parseFloat(document.getElementById('lotSize').value);
    const stopLoss = parseFloat(document.getElementById('stopLoss').value);
    const takeProfit = parseFloat(document.getElementById('takeProfit').value);
    const symbol = document.getElementById('orderSymbol').value;
    
    if (!lotSize || lotSize <= 0) {
        showToast('Invalid lot size', 'error');
        return;
    }
    
    const order = {
        symbol,
        direction,
        lot_size: lotSize,
        stop_loss_pips: stopLoss,
        take_profit_pips: takeProfit,
        confidence: 1.0 // Manual order
    };
    
    const result = await apiCall('/trade', 'POST', order);
    if (result) {
        showToast(`Order placed: ${direction} ${lotSize} ${symbol}`, 'success');
        loadPositions();
        updateAccountInfo();
    }
}

// Close All Positions
async function closeAllPositions() {
    if (!confirm('Are you sure you want to close all positions?')) return;
    
    const result = await apiCall('/positions/close_all', 'POST');
    if (result) {
        showToast('All positions closed', 'success');
        loadPositions();
        updateAccountInfo();
    }
}

// Trigger Kill Switch
async function triggerKillSwitch() {
    if (!confirm('WARNING: This will close all positions and stop trading. Continue?')) return;
    
    const result = await apiCall('/risk/kill_switch', 'POST');
    if (result) {
        showToast('Kill switch triggered - all positions closed', 'warning');
        updateKillSwitchStatus(true);
    }
}

// Toggle Mode (Paper/Live)
async function toggleMode() {
    const newMode = state.mode === 'paper' ? 'live' : 'paper';
    
    if (newMode === 'live') {
        const confirmed = confirm('WARNING: Switching to LIVE mode will use real money. Continue?');
        if (!confirmed) return;
    }
    
    const result = await apiCall('/mode', 'PUT', { mode: newMode });
    if (result) {
        state.mode = newMode;
        const modeBtn = document.getElementById('modeToggle');
        modeBtn.innerHTML = `<i class="fas fa-exchange-alt"></i> ${newMode === 'paper' ? 'Paper Mode' : 'LIVE Mode'}`;
        modeBtn.style.background = newMode === 'live' ? '#DC3545' : '';
        showToast(`Switched to ${newMode.toUpperCase()} mode`, 'warning');
    }
}

// Save Settings
async function saveSettings() {
    const settings = {
        max_risk_per_trade: parseFloat(document.getElementById('settingMaxRisk').value),
        max_concurrent_trades: parseInt(document.getElementById('settingMaxConcurrent').value),
        min_confidence: parseFloat(document.getElementById('settingMinConfidence').value),
        sl_atr_multiplier: parseFloat(document.getElementById('settingSLMultiplier').value),
        tp_atr_multiplier: parseFloat(document.getElementById('settingTPMultiplier').value),
        alerts: {
            on_trade: document.getElementById('alertOnTrade').checked,
            on_kill_switch: document.getElementById('alertOnKillSwitch').checked,
            on_drift: document.getElementById('alertOnDrift').checked
        },
        slack_webhook: document.getElementById('slackWebhook').value
    };
    
    const result = await apiCall('/settings', 'PUT', settings);
    if (result) {
        showToast('Settings saved successfully', 'success');
    }
}

// Reset Settings
function resetSettings() {
    document.getElementById('settingMaxRisk').value = '1.0';
    document.getElementById('settingMaxConcurrent').value = '3';
    document.getElementById('settingMinConfidence').value = '0.65';
    document.getElementById('settingSLMultiplier').value = '1.5';
    document.getElementById('settingTPMultiplier').value = '2.5';
    document.getElementById('alertOnTrade').checked = true;
    document.getElementById('alertOnKillSwitch').checked = true;
    document.getElementById('alertOnDrift').checked = true;
    document.getElementById('slackWebhook').value = '';
    
    showToast('Settings reset to default', 'info');
}

// Export Trade History
async function exportTradeHistory() {
    const trades = state.trades;
    if (!trades.length) {
        showToast('No trades to export', 'warning');
        return;
    }
    
    // Convert to CSV
    const headers = ['Time', 'Symbol', 'Direction', 'Entry', 'Exit', 'P&L', 'R Multiple', 'Duration'];
    const rows = trades.map(t => [
        t.close_time,
        t.symbol,
        t.direction,
        t.entry_price,
        t.exit_price,
        t.pnl,
        t.r_multiple,
        t.duration_minutes
    ]);
    
    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trading_history_${new Date().toISOString().slice(0, 19)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('Trade history exported', 'success');
}

// Render Functions
function renderSignalsTable(signals) {
    const tbody = document.querySelector('#signalsTable tbody');
    if (!tbody) return;
    
    if (!signals.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center">No signals available</td></tr>';
        return;
    }
    
    tbody.innerHTML = signals.slice(0, 20).map(signal => `
        <tr>
            <td>${new Date(signal.timestamp).toLocaleString()}</td>
            <td>${signal.symbol}</td>
            <td class="signal-${signal.direction.toLowerCase()}">${signal.direction}</td>
            <td>${(signal.confidence * 100).toFixed(1)}%</td>
            <td>
                <span title="Order Block: ${signal.smc_features?.dist_to_ob?.toFixed(2) || 'N/A'}">
                    OB: ${signal.smc_features?.has_ob ? '✓' : '✗'}
                </span>
                <span title="FVG: ${signal.smc_features?.has_fvg ? 'Present' : 'None'}">
                    FVG: ${signal.smc_features?.has_fvg ? '✓' : '✗'}
                </span>
            </td>
            <td>
                <span class="status-badge ${signal.status}">${signal.status}</span>
            </td>
        </tr>
    `).join('');
}

function renderPositionsTable(positions) {
    const tbody = document.querySelector('#positionsTable tbody');
    if (!tbody) return;
    
    if (!positions.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center">No open positions</td></tr>';
        return;
    }
    
    tbody.innerHTML = positions.map(pos => `
        <tr>
            <td>${pos.id?.slice(-8) || 'N/A'}</td>
            <td>${pos.symbol}</td>
            <td class="${pos.side.toLowerCase() === 'buy' ? 'signal-buy' : 'signal-sell'}">${pos.side}</td>
            <td>${pos.quantity}</td>
            <td>${pos.open_price.toFixed(3)}</td>
            <td>${pos.current_price.toFixed(3)}</td>
            <td class="${pos.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                ${pos.unrealized_pnl >= 0 ? '+' : ''}$${pos.unrealized_pnl.toFixed(2)}
            </td>
            <td>
                <button class="close-position-btn" data-id="${pos.id}">
                    <i class="fas fa-times"></i> Close
                </button>
            </td>
        </tr>
    `).join('');
    
    // Add close button handlers
    document.querySelectorAll('.close-position-btn').forEach(btn => {
        btn.addEventListener('click', () => closePosition(btn.dataset.id));
    });
}

function renderHistoryTable(trades) {
    const tbody = document.querySelector('#historyTable tbody');
    if (!tbody) return;
    
    if (!trades.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center">No trade history</td></tr>';
        return;
    }
    
    tbody.innerHTML = trades.slice(0, 100).map(trade => `
        <tr>
            <td>${new Date(trade.close_time).toLocaleString()}</td>
            <td>${trade.symbol}</td>
            <td class="${trade.direction.toLowerCase() === 'buy' ? 'signal-buy' : 'signal-sell'}">${trade.direction}</td>
            <td>${trade.entry_price.toFixed(3)}</td>
            <td>${trade.exit_price.toFixed(3)}</td>
            <td class="${trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                ${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}
            </td>
            <td>${trade.r_multiple?.toFixed(2) || 'N/A'}R</td>
            <td>${Math.floor(trade.duration_minutes / 60)}h ${trade.duration_minutes % 60}m</td>
        </tr>
    `).join('');
}

// Helper Functions
function updateConnectionStatus(connected) {
    state.connected = connected;
    const statusDiv = document.getElementById('connectionStatus');
    statusDiv.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
    statusDiv.innerHTML = `<i class="fas fa-circle"></i><span>${connected ? 'Connected' : 'Disconnected'}</span>`;
}

function updateMarketData(data) {
    state.marketData = data;
    document.getElementById('bidPrice').textContent = data.bid.toFixed(3);
    document.getElementById('askPrice').textContent = data.ask.toFixed(3);
    document.getElementById('spread').textContent = data.spread.toFixed(1);
}

function updateSignalDisplay(signal) {
    const signalDiv = document.getElementById('currentSignal');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    
    signalDiv.textContent = signal.direction;
    signalDiv.className = `signal-value ${signal.direction.toLowerCase()}`;
    
    const confidencePercent = signal.confidence * 100;
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceText.textContent = `${confidencePercent.toFixed(1)}% confidence`;
}

function addSignalToTable(signal) {
    const tbody = document.querySelector('#signalsTable tbody');
    if (!tbody) return;
    
    const newRow = document.createElement('tr');
    newRow.innerHTML = `
        <td>${new Date().toLocaleString()}</td>
        <td>${signal.symbol || 'XAUUSD'}</td>
        <td class="signal-${signal.direction.toLowerCase()}">${signal.direction}</td>
        <td>${(signal.confidence * 100).toFixed(1)}%</td>
        <td>OB: ${signal.has_ob ? '✓' : '✗'} | FVG: ${signal.has_fvg ? '✓' : '✗'}</td>
        <td><span class="status-badge pending">Pending</span></td>
    `;
    
    tbody.insertBefore(newRow, tbody.firstChild);
    if (tbody.children.length > 50) {
        tbody.removeChild(tbody.lastChild);
    }
}

function updateKillSwitchStatus(active) {
    const statusDiv = document.getElementById('killSwitchStatus');
    if (active) {
        statusDiv.innerHTML = '<span class="status-indicator inactive"></span><span>Kill Switch ACTIVE</span>';
    } else {
        statusDiv.innerHTML = '<span class="status-indicator active"></span><span>System Active</span>';
    }
}

function updateRiskInfo() {
    const lotSize = parseFloat(document.getElementById('lotSize').value) || 0;
    const stopLoss = parseFloat(document.getElementById('stopLoss').value) || 0;
    const takeProfit = parseFloat(document.getElementById('takeProfit').value) || 0;
    
    const accountBalance = state.accountInfo.balance || 10000;
    const riskPercent = 0.01; // 1%
    const riskAmount = accountBalance * riskPercent;
    
    const pipValue = 10; // $10 per pip for 1 lot on XAUUSD
    const slCost = stopLoss * pipValue * lotSize;
    const tpReward = takeProfit * pipValue * lotSize;
    
    const riskDisplay = slCost > 0 ? slCost : 0;
    const rr = slCost > 0 ? tpReward / slCost : 0;
    
    document.getElementById('riskInfo').innerHTML = `
        <span>Risk: $${riskDisplay.toFixed(2)} (${(riskDisplay / accountBalance * 100).toFixed(1)}%)</span>
        <span>Reward: $${tpReward.toFixed(2)}</span>
        <span>R:R: 1:${rr.toFixed(2)}</span>
    `;
}

function updatePnLDistribution(trades) {
    if (!state.charts.pnlDistribution) return;
    
    const bins = [-100, -50, 0, 50, 100, 150, 200];
    const counts = new Array(bins.length).fill(0);
    
    trades.forEach(trade => {
        const pnl = trade.pnl;
        let binIndex = bins.length - 1;
        for (let i = 0; i < bins.length; i++) {
            if (pnl <= bins[i]) {
                binIndex = i;
                break;
            }
        }
        counts[binIndex]++;
    });
    
    state.charts.pnlDistribution.data.datasets[0].data = counts;
    state.charts.pnlDistribution.update();
}

function updateEquityChart(period) {
    // Would fetch equity data for the selected period
    console.log(`Updating equity chart for period: ${period}`);
}

function changeChartInterval(interval) {
    document.querySelectorAll('.tool-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.interval === interval) {
            btn.classList.add('active');
        }
    });
    console.log(`Changing chart interval to: ${interval}`);
}

function toggleIndicator() {
    console.log('Indicator toggled');
}

async function closePosition(positionId) {
    const result = await apiCall(`/positions/${positionId}`, 'DELETE');
    if (result) {
        showToast('Position closed', 'success');
        loadPositions();
        updateAccountInfo();
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (state.updateInterval) {
        clearInterval(state.updateInterval);
    }
    if (state.socket) {
        state.socket.close();
    }
});
