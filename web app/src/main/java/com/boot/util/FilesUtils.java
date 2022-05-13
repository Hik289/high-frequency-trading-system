package com.boot.util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import lombok.extern.java.Log;

@Log
@Component
public class FilesUtils {
	// private static final String path = System.getProperty("user.dir") +
	// File.separator + "images" + File.separator;
	// public static final String path_file = System.getProperty("user.dir") +
	// File.separator + "files" + File.separator;

	private static String path = "";// 文件路径

	@Value("${file.path}")
	public void setPath(String path) {
		FilesUtils.path = path;
	}
	/**
	 * 保存文件或图片
	 * 
	 * @param file
	 * @param fileName
	 *            文件名 使用uuid
	 */
	public static String fileOut(HttpServletRequest request, MultipartFile file, String fileName) throws Exception {
		try {
			File uploadFilePath = new File(path);
			if (uploadFilePath.exists() == false) {
				uploadFilePath.mkdirs();
			}
			file.transferTo(new File(path + fileName));
		} catch (Exception e) {
			log.info(e.getMessage());
			throw new Exception("文件保存异常");
		}
		return path + fileName;
	}

	public static String fileUpload(HttpServletRequest request, MultipartFile file, String fileName, String path)
			throws Exception {
		try {
			File uploadFilePath = new File(path);
			if (uploadFilePath.exists() == false) {
				uploadFilePath.mkdirs();
			}
			file.transferTo(new File(path + fileName));
		} catch (Exception e) {
			log.info(e.getMessage());
			throw new Exception("文件保存异常");
		}
		return path + fileName;
	}

	public static void fileDownload(HttpServletResponse response, String path, String filepathname) throws Exception {
		File file = new File(path + filepathname);
		if (!file.exists()) {
			throw new Exception("文件不存在，请联系管理员！");
		}
		BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file));
		BufferedOutputStream bos = new BufferedOutputStream(response.getOutputStream());
		byte[] buf = new byte[1024];
		while (bis.read(buf, 0, buf.length) != -1) {
			bos.write(buf, 0, buf.length);
		}
		bis.close();
		bos.close();
	}

	/**
	 * 读取文件或图片
	 * @param response
	 * @param fileName
	 * @param type
	 * 类型 1 图片 2 文件
	 * @throws Exception
	 */
	public static void fileIn(HttpServletResponse response, String fileName, Integer type) throws Exception {
		try {
			File file = new File(path + fileName);
			if (!file.exists()) {
				throw new Exception("文件不存在，请联系管理员！");
			}
			response.setContentType("image/jpeg");
			response.setHeader("Content-Disposition", "inline;");
			if (type != null && type == 2) {
				response.setContentType("application/octet-stream");
				response.setHeader("Content-Disposition","attachment;filename=" + new String(fileName.getBytes("UTF-8"), "iso-8859-1"));
			}
			BufferedOutputStream bos = null;
			InputStream fis = null;
			BufferedInputStream bis = null;
			try {
				bos = new BufferedOutputStream(response.getOutputStream());
				fis = new FileInputStream(path + fileName);
				bis = new BufferedInputStream(fis);
				byte[] buff = new byte[2048];
				int bytesRead;
				while (-1 != (bytesRead = bis.read(buff, 0, buff.length))) {
					bos.write(buff, 0, bytesRead);
				}
				bos.flush();
			} catch (Exception e) {
				// e.printStackTrace();
				log.info(e.getMessage());
			} finally {
				if (bos != null) {
					bos.close();
				}
				if (bis != null) {
					bis.close();
				}
				if (fis != null) {
					fis.close();
				}
			}
		} catch (Exception e) {
			log.info(e.getMessage());
		}
	}
	// /**
	// * 移动端图片压缩
	// */
	// public static void weuiImageIn(HttpServletResponse response, String image)
	// throws Exception {
	// try {
	// response.setContentType("image/jpeg");
	// response.setHeader("Content-Disposition", "inline;");
	// BufferedOutputStream bos = null;
	// InputStream fis = null;
	// BufferedInputStream bis = null;
	// try {
	// bos = new BufferedOutputStream(response.getOutputStream());
	//
	// String linpath = path+UUID.randomUUID()+".jpg";
	// File distfile = new File(linpath);
	// //图片压缩0.5
	// Thumbnails.of(path+image).scale(1f).outputQuality(0.5f).toFile(linpath);
	// //fis = new FileInputStream(path+image);
	// fis = new FileInputStream(linpath);//使用临时压缩图片
	//
	// bis = new BufferedInputStream(fis);
	// byte[] buff = new byte[2048];
	// int bytesRead;
	// while (-1 != (bytesRead = bis.read(buff, 0, buff.length))) {
	// bos.write(buff, 0, bytesRead);
	// }
	// bos.flush();
	// bos.close();bis.close();fis.close();
	// //System.gc();
	// distfile.delete();//删除临时图片
	// } catch (Exception e) {
	// log.info(e.getMessage());
	// } finally {
	// if (bos != null) {
	// bos.close();
	// }
	// if (bis != null) {
	// bis.close();
	// }
	// if (fis != null) {
	// fis.close();
	// }
	// }
	// }catch (Exception e) {
	// log.info(e.getMessage());
	// }
	// }

	public static Integer isImg(String fileName, long fileSize) {
		String fileTyle = fileName.substring(fileName.lastIndexOf("."), fileName.length());
		if (!".jpg".equals(fileTyle) && !".jpeg".equals(fileTyle) && !".png".equals(fileTyle)
				&& !".gif".equals(fileTyle)) {
			return -1;
		}
		if (fileSize / 1024 > 1024) {
			return -2;
		}
		return 1;
	}
}