import stripe
from datetime import datetime
import os
from sqlalchemy.orm import Session
from database import User

# Configurar Stripe - En producción, usar variables de entorno
stripe.api_key = "tu_stripe_secret_key"  # Reemplazar con tu clave secreta de Stripe

# Planes de suscripción
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Plan Gratuito",
        "price_id": None,  # Plan gratuito no necesita price_id
        "features": [
            "3 comparaciones gratuitas",
            "Análisis básico",
            "Soporte por email"
        ],
        "price": "0",
        "interval": None,
        "limit": 3
    },
    "basic": {
        "name": "Plan Básico",
        "price_id": "price_XXXXX",  # Reemplazar con tu Price ID de Stripe
        "features": [
            "25 comparaciones por mes",
            "Análisis detallado",
            "Soporte por email",
            "Historial de comparaciones"
        ],
        "price": "9.99",
        "interval": "month",
        "limit": 25
    },
    "pro": {
        "name": "Plan Profesional",
        "price_id": "price_YYYYY",  # Reemplazar con tu Price ID de Stripe
        "features": [
            "Comparaciones ilimitadas",
            "Análisis avanzado",
            "Soporte prioritario 24/7",
            "API access",
            "Reportes personalizados"
        ],
        "price": "49.99",
        "interval": "month",
        "limit": float('inf')
    }
}

def create_checkout_session(plan_id: str, user_id: int, success_url: str, cancel_url: str):
    """Crear una sesión de checkout de Stripe"""
    if plan_id == "free":
        return None  # No crear checkout para plan gratuito
        
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': SUBSCRIPTION_PLANS[plan_id]['price_id'],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=str(user_id),
        )
        return checkout_session
    except Exception as e:
        raise Exception(f"Error creating checkout session: {str(e)}")

def get_subscription_status(user_id: int, db: Session):
    """Obtener el estado de suscripción de un usuario"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        # Verificar plan gratuito
        if not user.subscription_plan or user.subscription_plan == "free":
            remaining = 3 - (user.comparisons_count or 0)
            return {
                'status': 'active',
                'plan': 'free',
                'remaining_comparisons': max(0, remaining),
                'is_free_plan': True
            }
            
        if not user.stripe_customer_id:
            return None
        
        subscriptions = stripe.Subscription.list(
            customer=user.stripe_customer_id,
            status='active',
            limit=1
        )
        
        if not subscriptions.data:
            return None
            
        subscription = subscriptions.data[0]
        plan = subscription.plan.nickname or subscription.plan.id
        return {
            'status': subscription.status,
            'current_period_end': datetime.fromtimestamp(subscription.current_period_end),
            'plan': plan,
            'is_free_plan': False
        }
    except Exception as e:
        raise Exception(f"Error checking subscription status: {str(e)}")

def update_comparison_count(user_id: int, db: Session):
    """Actualizar el contador de comparaciones de un usuario"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.comparisons_count = (user.comparisons_count or 0) + 1
            user.last_comparison_date = datetime.utcnow()
            db.commit()
            return user.comparisons_count
    except Exception as e:
        db.rollback()
        raise Exception(f"Error updating comparison count: {str(e)}")

def can_make_comparison(user_id: int, db: Session) -> bool:
    """Verificar si un usuario puede realizar una comparación"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
            
        # Si no tiene plan, asignar plan gratuito
        if not user.subscription_plan:
            user.subscription_plan = "free"
            db.commit()
        
        # Verificar límites según el plan
        if user.subscription_plan == "free":
            return (user.comparisons_count or 0) < SUBSCRIPTION_PLANS["free"]["limit"]
        elif user.subscription_plan in SUBSCRIPTION_PLANS:
            return (user.comparisons_count or 0) < SUBSCRIPTION_PLANS[user.subscription_plan]["limit"]
            
        return False
    except Exception as e:
        return False

def cancel_subscription(subscription_id: str):
    """Cancelar una suscripción"""
    try:
        subscription = stripe.Subscription.delete(subscription_id)
        return subscription
    except Exception as e:
        raise Exception(f"Error canceling subscription: {str(e)}")

def handle_webhook(payload, sig_header, webhook_secret):
    """Manejar webhooks de Stripe"""
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
        
        # Manejar diferentes eventos
        if event.type == 'customer.subscription.created':
            # Actualizar estado de suscripción en la base de datos
            pass
        elif event.type == 'customer.subscription.deleted':
            # Actualizar estado de suscripción en la base de datos
            pass
        
        return event
        
    except Exception as e:
        raise Exception(f"Error handling webhook: {str(e)}") 