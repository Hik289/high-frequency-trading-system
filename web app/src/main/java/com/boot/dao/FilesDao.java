package com.boot.dao;

import java.util.List;

import org.apache.ibatis.annotations.Param;

import com.boot.entity.Files;

public interface FilesDao {

	List<Files> getListByCard(@Param("cardId")Integer cardId);

	void deletByCard(@Param("cardId")Integer cardId);

	void addByCard(@Param("cardId")Integer cardId,@Param("fileId")String[] fileId);

}
