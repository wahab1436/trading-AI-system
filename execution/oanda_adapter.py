"""OANDA REST API broker adapter implementation"""

import logging
import requests
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time
import hashlib
import hmac

from .broker_adapter import (
    BrokerAdapter, Order, Position, AccountInfo,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class OANDAAdapter(BrokerAdapter):
    """OANDA REST API v20 implementation"""
    
    def __init__(
        self,
        api_key: str = None,
        account_id: str = None,
        environment: str = "practice",  # or "live"
        host: str = None
    ):
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        
        # Set host based on environment
        if host:
            self.host = host
        elif environment == "practice":
            self.host = "api-fxpractice.oanda.com"
        else:
            self.host = "api-fxtrade.oanda.com"
            
        self.base_url = f"https://{self.host}/v3"
        self.stream_url = f"https://stream-fxpractice.oanda.com/v3" if environment == "practice" else f"https://stream-fxtrade.oanda.com/v3"
        
        self._connected = False
        self._session = None
        self._stream_session = None
        
        # Account and instrument details
        self.account_currency = "USD"
        self.instruments = {}
        self.pricing_info = {}
        
        # Rate limiting
        self.requests_per_second = 5
        self.last_request_time = 0
        
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None,
        retry_count: int = 3
    ) -> Optional[Dict]:
        """Make authenticated request to OANDA API with rate limiting"""
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < 1.0 / self.requests_per_second:
            time.sleep((1.0 / self.requests_per_second) - elapsed)
            
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }
        
        for attempt in range(retry_count):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=30
                )
                
                self.last_request_time = time.time()
                
                if response.status_code == 200 or response.status_code == 201:
                    return response.json()
                elif response.status_code == 401:
                    logger.error("OANDA authentication failed - invalid API key")
                    return None
                elif response.status_code == 429:
                    # Rate limit exceeded - wait and retry
                    wait_time = int(response.headers.get('X-RateLimit-Reset', 60))
                    logger.warning(f"Rate limit exceeded, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"OANDA API error {response.status_code}: {response.text}")
                    if attempt == retry_count - 1:
                        return None
                        
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
                
        return None
        
    def _make_stream_request(self, endpoint: str) -> Optional[requests.Response]:
        """Make streaming request to OANDA API"""
        
        url = f"{self.stream_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }
        
        try:
            response = self._stream_session.get(url, headers=headers, stream=True, timeout=60)
            if response.status_code == 200:
                return response
            else:
                logger.error(f"Stream request failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Stream request error: {e}")
            return None
            
    def connect(self) -> bool:
        """Connect to OANDA API"""
        
        if not self.api_key or not self.account_id:
            logger.error("API key and Account ID are required")
            return False
            
        self._session = requests.Session()
        self._stream_session = requests.Session()
        
        # Test connection by getting account summary
        account_info = self._make_request("GET", f"/accounts/{self.account_id}/summary")
        
        if account_info:
            self._connected = True
            self.account_currency = account_info.get('account', {}).get('currency', 'USD')
            logger.info(f"Connected to OANDA {self.environment} - Account: {self.account_id}")
            
            # Load instrument details
            self._load_instruments()
            return True
            
        return False
        
    def _load_instruments(self):
        """Load available instruments and their details"""
        
        response = self._make_request("GET", f"/accounts/{self.account_id}/instruments")
        
        if response and 'instruments' in response:
            for inst in response['instruments']:
                self.instruments[inst['name']] = {
                    'display_name': inst['displayName'],
                    'pip_location': inst.get('pipLocation', -4),
                    'trade_units_precision': inst.get('tradeUnitsPrecision', 0),
                    'minimum_trade_size': inst.get('minimumTradeSize', '1'),
                    'maximum_trade_size': inst.get('maximumTradeSize', '10000000'),
                    'margin_rate': float(inst.get('marginRate', 0.02))
                }
            logger.info(f"Loaded {len(self.instruments)} instruments")
            
    def disconnect(self) -> bool:
        """Disconnect from OANDA API"""
        
        if self._session:
            self._session.close()
        if self._stream_session:
            self._stream_session.close()
            
        self._connected = False
        logger.info("Disconnected from OANDA")
        return True
        
    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connected and self._session is not None
        
    def place_order(self, order: Order) -> Order:
        """Place an order on OANDA"""
        
        if not self.is_connected():
            logger.error("Not connected to OANDA")
            order.status = OrderStatus.REJECTED
            return order
            
        # Map order type to OANDA format
        order_data = self._build_order_request(order)
        
        response = self._make_request(
            "POST",
            f"/accounts/{self.account_id}/orders",
            data=order_data
        )
        
        if not response:
            order.status = OrderStatus.REJECTED
            return order
            
        # Parse response
        if 'orderCreateTransaction' in response:
            tx = response['orderCreateTransaction']
            order.id = tx.get('id', '')
            order.status = self._map_oanda_order_state(tx.get('type'))
            
            # If market order, get fill details
            if order.type == OrderType.MARKET and 'orderFillTransaction' in response:
                fill_tx = response['orderFillTransaction']
                order.status = OrderStatus.FILLED
                order.filled_quantity = float(fill_tx.get('units', 0))
                order.average_price = float(fill_tx.get('price', 0))
                order.filled_at = self._parse_oanda_time(fill_tx.get('time'))
                
            logger.info(f"Order placed: {order.id} - {order.side.value} {order.quantity} {order.symbol}")
            
        elif 'orderRejectTransaction' in response:
            reject = response['orderRejectTransaction']
            logger.error(f"Order rejected: {reject.get('rejectReason', 'Unknown')}")
            order.status = OrderStatus.REJECTED
            
        return order
        
    def _build_order_request(self, order: Order) -> Dict:
        """Build OANDA order request"""
        
        # Convert quantity to OANDA units
        units = self._lot_to_units(order.symbol, order.quantity)
        
        if order.type == OrderType.MARKET:
            return {
                "order": {
                    "type": "MARKET",
                    "instrument": order.symbol,
                    "units": str(units if order.side == OrderSide.BUY else -units),
                    "timeInForce": "FOK"  # Fill or Kill
                }
            }
        elif order.type == OrderType.LIMIT:
            return {
                "order": {
                    "type": "LIMIT",
                    "instrument": order.symbol,
                    "units": str(units if order.side == OrderSide.BUY else -units),
                    "price": str(order.price),
                    "timeInForce": "GTC",
                    "positionFill": "DEFAULT"
                }
            }
        elif order.type == OrderType.STOP:
            return {
                "order": {
                    "type": "STOP",
                    "instrument": order.symbol,
                    "units": str(units if order.side == OrderSide.BUY else -units),
                    "price": str(order.price),
                    "timeInForce": "GTC",
                    "positionFill": "DEFAULT"
                }
            }
        else:
            raise ValueError(f"Unsupported order type: {order.type}")
            
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        
        if not self.is_connected():
            return False
            
        response = self._make_request(
            "PUT",
            f"/accounts/{self.account_id}/orders/{order_id}/cancel"
        )
        
        success = response is not None and 'orderCancelTransaction' in response
        
        if success:
            logger.info(f"Order {order_id} cancelled")
        else:
            logger.error(f"Failed to cancel order {order_id}")
            
        return success
        
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """Modify a pending order (price, units, SL, TP)"""
        
        if not self.is_connected():
            return False
            
        modify_data = {
            "order": {
                "type": "LIMIT"  # OANDA requires specifying order type
            }
        }
        
        if 'price' in kwargs:
            modify_data['order']['price'] = str(kwargs['price'])
        if 'stop_loss' in kwargs:
            modify_data['order']['stopLossOnFill'] = {
                "price": str(kwargs['stop_loss'])
            }
        if 'take_profit' in kwargs:
            modify_data['order']['takeProfitOnFill'] = {
                "price": str(kwargs['take_profit'])
            }
            
        response = self._make_request(
            "PUT",
            f"/accounts/{self.account_id}/orders/{order_id}",
            data=modify_data
        )
        
        return response is not None
        
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        
        if not self.is_connected():
            return None
            
        response = self._make_request(
            "GET",
            f"/accounts/{self.account_id}/orders/{order_id}"
        )
        
        if not response:
            return None
            
        # OANDA returns order in 'order' field
        oanda_order = response.get('order', {})
        
        return Order(
            id=oanda_order.get('id', ''),
            symbol=oanda_order.get('instrument', ''),
            side=OrderSide.BUY if float(oanda_order.get('units', 0)) > 0 else OrderSide.SELL,
            type=self._map_oanda_order_type(oanda_order.get('type', '')),
            quantity=self._units_to_lot(oanda_order.get('instrument', ''), abs(float(oanda_order.get('units', 0)))),
            price=float(oanda_order.get('price', 0)) if oanda_order.get('price') else None,
            stop_loss=float(oanda_order.get('stopLossOnFill', {}).get('price', 0)) if oanda_order.get('stopLossOnFill') else None,
            take_profit=float(oanda_order.get('takeProfitOnFill', {}).get('price', 0)) if oanda_order.get('takeProfitOnFill') else None,
            status=self._map_oanda_order_state(oanda_order.get('state', '')),
            created_at=self._parse_oanda_time(oanda_order.get('createTime'))
        )
        
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        
        if not self.is_connected():
            return []
            
        response = self._make_request(
            "GET",
            f"/accounts/{self.account_id}/positions"
        )
        
        if not response:
            return []
            
        positions = []
        for pos in response.get('positions', []):
            # Check long position
            if 'long' in pos and float(pos['long'].get('units', 0)) != 0:
                positions.append(self._parse_position(pos['instrument'], pos['long'], OrderSide.BUY))
                
            # Check short position
            if 'short' in pos and float(pos['short'].get('units', 0)) != 0:
                positions.append(self._parse_position(pos['instrument'], pos['short'], OrderSide.SELL))
                
        return positions
        
    def _parse_position(self, symbol: str, pos_data: Dict, side: OrderSide) -> Position:
        """Parse OANDA position data"""
        
        # Get current price for P&L calculation
        current_price = self._get_current_price(symbol)
        
        open_price = float(pos_data.get('averagePrice', 0))
        units = abs(float(pos_data.get('units', 0)))
        quantity = self._units_to_lot(symbol, units)
        unrealized_pl = float(pos_data.get('unrealizedPL', 0))
        
        return Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            open_price=open_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pl,
            realized_pnl=0,
            stop_loss=float(pos_data.get('stopLossOrder', {}).get('price', 0)) if pos_data.get('stopLossOrder') else None,
            take_profit=float(pos_data.get('takeProfitOrder', {}).get('price', 0)) if pos_data.get('takeProfitOrder') else None,
            open_time=self._parse_oanda_time(pos_data.get('openTime'))
        )
        
    def close_position(self, position_id: str) -> bool:
        """Close a position (OANDA closes by instrument, not position ID)"""
        
        if not self.is_connected():
            return False
            
        # For OANDA, we need the instrument symbol
        # position_id here is actually the instrument symbol
        response = self._make_request(
            "PUT",
            f"/accounts/{self.account_id}/positions/{position_id}/close",
            data={"longUnits": "ALL", "shortUnits": "ALL"}
        )
        
        success = response is not None and 'orderFillTransaction' in response
        
        if success:
            logger.info(f"Position {position_id} closed")
            
        return success
        
    def get_account_info(self) -> AccountInfo:
        """Get account information"""
        
        if not self.is_connected():
            return AccountInfo(balance=0, equity=0, margin=0, free_margin=0, margin_level=0)
            
        response = self._make_request(
            "GET",
            f"/accounts/{self.account_id}/summary"
        )
        
        if not response:
            return AccountInfo(balance=0, equity=0, margin=0, free_margin=0, margin_level=0)
            
        account = response.get('account', {})
        
        return AccountInfo(
            balance=float(account.get('balance', 0)),
            equity=float(account.get('NAV', 0)),
            margin=float(account.get('marginUsed', 0)),
            free_margin=float(account.get('marginAvailable', 0)),
            margin_level=float(account.get('marginRate', 0)) * 100 if account.get('marginRate') else 0,
            currency=account.get('currency', 'USD'),
            unrealized_pnl=float(account.get('unrealizedPL', 0)),
            realized_pnl_today=float(account.get('pl', 0))
        )
        
    def get_ticks(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get recent ticks/candles"""
        
        if not self.is_connected():
            return []
            
        response = self._make_request(
            "GET",
            f"/accounts/{self.account_id}/instruments/{symbol}/candles",
            params={
                "count": count,
                "granularity": "S5",  # 5-second candles
                "price": "MBA"  # Mid, Bid, Ask
            }
        )
        
        if not response:
            return []
            
        ticks = []
        for candle in response.get('candles', []):
            if candle.get('complete'):
                mid = candle.get('mid', {})
                ticks.append({
                    'time': self._parse_oanda_time(candle.get('time')),
                    'open': float(mid.get('o', 0)),
                    'high': float(mid.get('h', 0)),
                    'low': float(mid.get('l', 0)),
                    'close': float(mid.get('c', 0)),
                    'volume': candle.get('volume', 0)
                })
                
        return ticks
        
    def stream_prices(self, symbols: List[str], callback):
        """Stream live prices via WebSocket (OANDA uses streaming API)"""
        
        if not self.is_connected():
            logger.error("Not connected to OANDA")
            return
            
        # OANDA streaming endpoint
        stream_response = self._make_stream_request(f"/accounts/{self.account_id}/pricing/stream?instruments={','.join(symbols)}")
        
        if not stream_response:
            logger.error("Failed to connect to price stream")
            return
            
        for line in stream_response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'type' in data and data['type'] == 'PRICE':
                        self._process_stream_price(data, callback)
                except Exception as e:
                    logger.error(f"Error processing stream data: {e}")
                    
    def _process_stream_price(self, price_data: Dict, callback):
        """Process streaming price data"""
        
        tick = {
            'symbol': price_data.get('instrument'),
            'bid': float(price_data.get('bids', [{}])[0].get('price', 0)) if price_data.get('bids') else 0,
            'ask': float(price_data.get('asks', [{}])[0].get('price', 0)) if price_data.get('asks') else 0,
            'timestamp': self._parse_oanda_time(price_data.get('time')),
            'spread': float(price_data.get('closeoutBid', 0)) - float(price_data.get('closeoutAsk', 0))
        }
        
        if callback:
            callback(tick)
            
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        
        response = self._make_request(
            "GET",
            f"/accounts/{self.account_id}/pricing",
            params={"instruments": symbol}
        )
        
        if response and response.get('prices'):
            price = response['prices'][0]
            return float(price.get('closeoutAsk', 0))
            
        return 0.0
        
    def _lot_to_units(self, symbol: str, lots: float) -> int:
        """Convert lots to OANDA units"""
        
        # Standard lot = 100,000 units
        # Mini lot = 10,000 units
        # Micro lot = 1,000 units
        
        units = int(lots * 100000)
        
        # Check instrument-specific constraints
        if symbol in self.instruments:
            min_units = int(float(self.instruments[symbol]['minimum_trade_size']))
            units = max(units, min_units)
            
        return units
        
    def _units_to_lot(self, symbol: str, units: float) -> float:
        """Convert OANDA units to lots"""
        
        return units / 100000
        
    def _map_oanda_order_type(self, oanda_type: str) -> OrderType:
        """Map OANDA order type to internal enum"""
        
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP": OrderType.STOP,
            "STOP_LOSS": OrderType.STOP,
            "TAKE_PROFIT": OrderType.LIMIT
        }
        return mapping.get(oanda_type, OrderType.MARKET)
        
    def _map_oanda_order_state(self, state: str) -> OrderStatus:
        """Map OANDA order state to internal enum"""
        
        mapping = {
            "PENDING": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "TRIGGERED": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }
        return mapping.get(state, OrderStatus.PENDING)
        
    def _parse_oanda_time(self, time_str: str) -> datetime:
        """Parse OANDA time string to datetime"""
        
        if not time_str:
            return datetime.utcnow()
            
        # OANDA uses RFC3339 format: 2024-01-15T10:30:00.000000000Z
        try:
            # Remove timezone info and parse
            time_str = time_str.replace('Z', '+00:00')
            return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except:
            return datetime.utcnow()
            
    def get_order_history(
        self,
        from_time: datetime = None,
        to_time: datetime = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get order history for analysis"""
        
        if not self.is_connected():
            return []
            
        params = {"count": limit}
        
        if from_time:
            params['from'] = from_time.isoformat()
        if to_time:
            params['to'] = to_time.isoformat()
            
        response = self._make_request(
            "GET",
            f"/accounts/{self.account_id}/orders",
            params=params
        )
        
        if not response:
            return []
            
        orders = []
        for order in response.get('orders', []):
            orders.append({
                'id': order.get('id'),
                'instrument': order.get('instrument'),
                'type': order.get('type'),
                'units': order.get('units'),
                'price': order.get('price'),
                'state': order.get('state'),
                'time': self._parse_oanda_time(order.get('createTime'))
            })
            
        return orders
        
    def get_trade_history(
        self,
        from_time: datetime = None,
        to_time: datetime = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get trade (fill) history"""
        
        if not self.is_connected():
            return []
            
        params = {"count": limit}
        
        if from_time:
            params['from'] = from_time.isoformat()
        if to_time:
            params['to'] = to_time.isoformat()
            
        response = self._make_request(
            "GET",
            f"/accounts/{self.account_id}/trades",
            params=params
        )
        
        if not response:
            return []
            
        trades = []
        for trade in response.get('trades', []):
            trades.append({
                'id': trade.get('id'),
                'instrument': trade.get('instrument'),
                'units': trade.get('units'),
                'price': trade.get('price'),
                'realizedPL': trade.get('realizedPL'),
                'openTime': self._parse_oanda_time(trade.get('openTime')),
                'closeTime': self._parse_oanda_time(trade.get('closeTime')) if trade.get('closeTime') else None
            })
            
        return trades
        
    def get_market_hours(self, symbol: str) -> Dict:
        """Get market trading hours for symbol"""
        
        # OANDA markets are 24/5 for forex, 24/6 for gold
        if symbol == "XAUUSD":
            # Gold: Sunday 23:00 - Friday 22:00 UTC
            return {
                'open_day': 6,  # Sunday
                'open_hour': 23,
                'close_day': 5,  # Friday
                'close_hour': 22,
                'is_24h': False
            }
        else:
            # Forex: Sunday 21:00 - Friday 22:00 UTC
            return {
                'open_day': 6,
                'open_hour': 21,
                'close_day': 5,
                'close_hour': 22,
                'is_24h': False
            }
            
    def validate_connection(self) -> Dict:
        """Validate connection and return status"""
        
        if not self.is_connected():
            return {'connected': False, 'error': 'Not connected'}
            
        # Test with account summary
        response = self._make_request("GET", f"/accounts/{self.account_id}/summary")
        
        if response:
            return {
                'connected': True,
                'account_id': self.account_id,
                'environment': self.environment,
                'balance': float(response.get('account', {}).get('balance', 0))
            }
        else:
            return {'connected': False, 'error': 'API request failed'}
