const $ = require('shelljs')
require('../logging')

$.config.fatal = true
const root = `${__dirname}/../..`


module.exports.build_image = function () {
    const image_name = 'zevere/nlp:latest'
    console.info(`building Docker Image "${image_name}"`)
    $.exec(`docker build -t ${image_name} ${root}/`)
}
