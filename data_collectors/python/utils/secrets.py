from logging import exception
from easysettings import JSONSettings, preferred_file
import os

SECRETS_PATH = "utils/appsettings.secrets.json"


def _get_user_secrets():
    if os.path.exists(SECRETS_PATH):
        return JSONSettings.from_file(
            preferred_file(
                [
                    SECRETS_PATH,
                ]
            )
        ).__dict__["data"]
    else:
        raise exception(
            f"\n\n-------\nERROR:\n    Usersecrets are used in this project, but the file has not been made locally.\n    Please create the file '{SECRETS_PATH}' and input all required secrets and try agian.\n------\n\n"
        )


def _get_attribute(attribute: str) -> str:
    secrets = _get_user_secrets()

    if attribute in secrets:
        return secrets[attribute]
    else:
        raise exception(
            f"\n\n------\nERROR:\n    The '{SECRETS_PATH}' file does not contain the requested attribute '{attribute}'\n    Please insert this attribute and try again.\n------\n\n"
        )


def get_database():
    return _get_attribute("database")


def get_user():
    return _get_attribute("user")


def get_pasword():
    return _get_attribute("password")


if __name__ == "__main__":
    database = get_database()
    user = get_user()
    password = get_pasword()

    print(
        f"\n\nUser secrets include:\n    - database: {database}\n    - user: {user}\n    - password: {password}\n\n"
    )
