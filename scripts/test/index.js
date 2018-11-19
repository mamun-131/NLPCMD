const $ = require('shelljs');
const path = require('path');
require('../logging')

$.config.fatal = true


try {
    console.warn("ToDo")
} catch (e) {
    console.error(e)
    process.exit(1)
}
