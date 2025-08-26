import csv
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()


def _get_env(name: str) -> str:
    """Return stripped environment variable without surrounding quotes."""
    return os.getenv(name, "").strip().strip('"')


def _normalize_shop_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.netloc


SHOP_URL = _normalize_shop_url(_get_env("SHOP_URL"))
ACCESS_TOKEN = _get_env("ACCESS_TOKEN")

API_VER = "2024-04"
MAX_RETRIES = 5

INVENTORY_CSV_URL = "https://haendler.spalex.de/SpaLeXBestandAbgleich.csv"

STOCK_MAPPING = {"0": 0, "1": 10}


class ShopifyAPIError(Exception):
    """Raised when the Shopify API request ultimately fails."""

    def __init__(self, message: str, response: Optional[requests.Response] = None) -> None:
        super().__init__(message)
        self.response = response


class RateLimiter:
    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    def wait(self) -> None:
        while True:
            now = time.time()
            while self.calls and now - self.calls[0] >= self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            sleep_time = self.period - (now - self.calls[0])
            time.sleep(max(sleep_time, 0))


def shopify_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"https://{SHOP_URL}/admin/api/{API_VER}/{path}"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": ACCESS_TOKEN,
    }

    last_response: Optional[requests.Response] = None
    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            resp = requests.request(method, url, headers=headers, **kwargs)
            last_response = resp
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = 2.0
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        try:
                            wait_dt = parsedate_to_datetime(retry_after)
                            wait = (
                                wait_dt - datetime.now(timezone.utc)
                            ).total_seconds()
                        except (TypeError, ValueError):
                            wait = 2.0
                logging.warning(
                    "⏰ Rate limit (429) bereikt voor %s, wachten %.2fs (poging %s/%s)",
                    url,
                    wait,
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(max(wait, 0))
                continue
            return resp
        except requests.exceptions.RequestException as exc:
            logging.error(
                "❌ Verbindingsfout tijdens aanvraag %s: %s (poging %s/%s)",
                url,
                exc,
                attempt,
                MAX_RETRIES,
            )
            if attempt < MAX_RETRIES:
                time.sleep(2**attempt)
                continue
            raise ShopifyAPIError(
                f"Aanvraag naar {url} mislukt na {MAX_RETRIES} pogingen", last_response
            ) from exc
    raise ShopifyAPIError(
        f"Aanvraag naar {url} mislukt na {MAX_RETRIES} pogingen", last_response
    )


def get_inventory_csv() -> pd.DataFrame:
    logging.info("Voorraadgegevens ophalen van INVENTORY_CSV_URL: %s", INVENTORY_CSV_URL)
    try:
        resp = requests.get(INVENTORY_CSV_URL, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logging.error("\u274c CSV download mislukt: %s", exc)
        return pd.DataFrame()

    encoding = resp.encoding or resp.apparent_encoding or "utf-8"
    try:
        content = resp.content.decode(encoding, errors="replace")
        reader = csv.DictReader(content.splitlines(), delimiter=";")
        df = pd.DataFrame(reader).fillna("")
        df.columns = df.columns.str.strip()
        df = df.rename(
            columns={"EAN/BARCODE": "EAN", "Lagerbestand(0,1)": "Voorraadniveau"}
        )

        required_columns = ["Artikelnummer", "EAN", "Voorraadniveau"]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            logging.error("\u274c Ontbrekende kolommen in CSV: %s", ", ".join(missing))
            return pd.DataFrame()

        if "Aktiv" in df.columns:
            before = len(df)
            df = df[df["Aktiv"].str.strip().str.upper() == "Y"]
            skipped = before - len(df)
            if skipped:
                logging.info("Skipped %s inactieve rijen (Aktiv)", skipped)
        elif "Lagervorhanden(J/N)" in df.columns:
            before = len(df)
            df = df[df["Lagervorhanden(J/N)"].str.strip().str.upper() == "J"]
            skipped = before - len(df)
            if skipped:
                logging.info(
                    "Skipped %s inactieve rijen (Lagervorhanden(J/N))",
                    skipped,
                )
        return df
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Fout bij het verwerken van de voorraad CSV: %s", exc)
        return pd.DataFrame()


def get_location_id() -> Optional[str]:
    logging.info("Locatie-ID ophalen...")
    try:
        r = shopify_request("GET", "locations.json")
    except ShopifyAPIError as exc:
        logging.error("\u274c Fout bij ophalen locaties: %s", exc)
        return None
    if r.status_code == 200:
        data = r.json()
        locations = data.get("locations") or []
        if not locations:
            logging.error("\u274c Geen locaties gevonden in response: %s", data)
            return None
        for loc in locations:
            if "spalex" in loc.get("name", "").lower():
                logging.info("\u2705 Locatie 'spalex' gevonden: %s (%s)", loc["name"], loc["id"])
                return str(loc["id"])
        logging.info(
            "Geen 'spalex' locatie gevonden, de eerste locatie wordt gebruikt: %s (%s)",
            locations[0]["name"],
            locations[0]["id"],
        )
        return str(locations[0]["id"])
    logging.error("\u274c Fout bij ophalen locaties: Status %s - %s", r.status_code, r.text)
    return None


def _search_variant_by_param(param_name: str, param_value: str) -> Optional[dict]:
    params = {param_name: param_value, "limit": 1}
    try:
        resp = shopify_request("GET", "variants.json", params=params)
        logging.info(
            "API respons voor %s '%s': Status %s",
            param_name.upper(),
            param_value,
            resp.status_code,
        )
    except ShopifyAPIError as exc:
        logging.warning(
            "Fout bij zoeken variant op %s '%s': %s",
            param_name.upper(),
            param_value,
            exc,
        )
        return None
    if resp.status_code == 200:
        variants = resp.json().get("variants", [])
        return variants[0] if variants else None
    if resp.status_code != 404:
        logging.warning(
            "Fout bij zoeken variant op %s '%s': Status %s - %s",
            param_name.upper(),
            param_value,
            resp.status_code,
            resp.text[:200],
        )
    return None


def get_variant_by_sku_or_ean(sku: str, ean: str) -> Optional[dict]:
    if not sku and not ean:
        logging.warning("SKU en EAN ontbreken, overslaan")
        return None

    logging.info("\U0001f50d Zoeken naar variant voor SKU '%s' of EAN '%s'", sku, ean)
    if sku:
        variant = _search_variant_by_param("sku", sku)
        if variant:
            logging.info("\u2705 Variant gevonden op SKU '%s': ID %s", sku, variant["id"])
            return variant
        logging.info("Geen variant gevonden op SKU '%s'", sku)

    if ean:
        variant = _search_variant_by_param("barcode", ean)
        if variant:
            logging.info("\u2705 Variant gevonden op EAN '%s': ID %s", ean, variant["id"])
            return variant
        logging.info("Geen variant gevonden op EAN '%s'", ean)

    logging.warning("\u274c Geen variant gevonden voor SKU '%s' of EAN '%s'", sku, ean)
    return None


def update_shopify_inventory(
    inventory_item_id: str, location_id: str, quantity: int, sku: str
) -> bool:
    payload = {
        "location_id": location_id,
        "inventory_item_id": inventory_item_id,
        "available": quantity,
    }
    try:
        resp = shopify_request("POST", "inventory_levels/set.json", json=payload)
    except ShopifyAPIError as exc:
        logging.error(
            "\u274c Fout bij instellen voorraad voor SKU '%s' (Item ID: %s): %s",
            sku,
            inventory_item_id,
            exc,
        )
        return False
    if resp.status_code == 200:
        logging.info(
            "\ud83d\udce6 Voorraad voor SKU '%s' (Item ID: %s) succesvol ingesteld op %s.",
            sku,
            inventory_item_id,
            quantity,
        )
        return True
    logging.error(
        "\u274c Fout bij instellen voorraad voor SKU '%s' (Item ID: %s): Status %s - %s",
        sku,
        inventory_item_id,
        resp.status_code,
        resp.text[:200],
    )
    return False


def normalize_stock(value: str) -> Optional[str]:
    """Return standardized stock value from raw CSV cell."""
    cleaned = value.strip().replace(" ", "").replace(",", ".").lower()
    try:
        num = float(cleaned)
    except ValueError:
        return None
    if num == 1.0:
        return "1"
    if num == 0.0:
        return "0"
    return None


def main() -> None:
    logging.info("Start Shopify voorraad updater script...")

    if not SHOP_URL or not ACCESS_TOKEN:
        logging.error(
            "Omgevingsvariabelen SHOP_URL en ACCESS_TOKEN zijn verplicht. Stop script."
        )
        return

    inventory_df = get_inventory_csv()
    if inventory_df.empty:
        logging.error("\u274c Geen geldige voorraadgegevens om te verwerken. Script voltooid.")
        return

    location_id = get_location_id()
    if not location_id:
        logging.error(
            "\u274c Stop: Geen geldige voorraadlocatie gevonden. Voorraadupdate afgebroken."
        )
        return

    update_log = []
    processed_count = updated_count = skipped_count = failed_count = 0

    for row in inventory_df.itertuples(index=False):
        try:
            artikelnummer = row.Artikelnummer.strip()
            ean = row.EAN.strip()
            voorraadniveau_raw = row.Voorraadniveau.strip()
            artikelnaam = getattr(row, "Artikelname", "").strip()

            norm = normalize_stock(voorraadniveau_raw)
            logging.info(
                "Verwerken: Artikelnummer '%s', EAN '%s', Voorraadniveau '%s' (genormaliseerd: '%s')",
                artikelnummer,
                ean,
                voorraadniveau_raw,
                norm,
            )
            if norm is None:
                logging.warning(
                    "\u26a0\ufe0f Onbekend 'Voorraadniveau' '%s' voor Artikelnummer '%s'. Overslaan.",
                    voorraadniveau_raw,
                    artikelnummer,
                )
                skipped_count += 1
                update_log.append(
                    {
                        "Artikelnummer": artikelnummer,
                        "EAN": ean,
                        "Artikelnaam": artikelnaam,
                        "CSV_Voorraadniveau": voorraadniveau_raw,
                        "Shopify_Variant_ID": "",
                        "Shopify_Item_ID": "",
                        "Target_Voorraad": "",
                        "Status": "Skipped (Onbekend Voorraadniveau)",
                        "Bericht": f"Onbekend 'Voorraadniveau': {voorraadniveau_raw}",
                    }
                )
                continue

            target_quantity = STOCK_MAPPING[norm]
            variant = get_variant_by_sku_or_ean(artikelnummer, ean)
            if variant:
                inventory_item_id = variant["inventory_item_id"]
                variant_id = variant["id"]
                success = update_shopify_inventory(
                    inventory_item_id, location_id, target_quantity, artikelnummer
                )
                if success:
                    updated_count += 1
                    status = "Succesvol bijgewerkt"
                    message = f"Voorraad ingesteld op {target_quantity}"
                else:
                    failed_count += 1
                    status = "Mislukt (API fout)"
                    message = "Fout bij Shopify API-aanroep"
            else:
                failed_count += 1
                variant_id = ""
                inventory_item_id = ""
                status = "Mislukt (Variant niet gevonden)"
                message = (
                    f"Geen overeenkomende variant gevonden voor SKU '{artikelnummer}' of EAN '{ean}'"
                )

            update_log.append(
                {
                    "Artikelnummer": artikelnummer,
                    "EAN": ean,
                    "Artikelnaam": artikelnaam,
                    "CSV_Voorraadniveau": voorraadniveau_raw,
                    "Shopify_Variant_ID": variant_id,
                    "Shopify_Item_ID": inventory_item_id,
                    "Target_Voorraad": target_quantity,
                    "Status": status,
                    "Bericht": message,
                }
            )
            processed_count += 1
        except Exception as exc:  # pylint: disable=broad-except
            failed_count += 1
            logging.exception("\u274c Onverwachte fout bij verwerken van rij: %s", exc)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"inventory_update_log_{timestamp}.csv"
    pd.DataFrame(update_log).to_csv(log_filename, index=False)

    logging.info("\n--- Voorraadupdate voltooid ---")
    logging.info("Totaal verwerkte items: %s", processed_count)
    logging.info("Aantal succesvol bijgewerkt: %s", updated_count)
    logging.info("Aantal overgeslagen items: %s", skipped_count)
    logging.info("Aantal mislukte updates: %s", failed_count)
    logging.info("Gedetailleerde log is opgeslagen in '%s'.", log_filename)


if __name__ == "__main__":
    limiter = RateLimiter(max_calls=2, period=1.0)
    main()
